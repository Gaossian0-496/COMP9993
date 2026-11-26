import argparse
import yaml
import os
import sys
sys.path.append(os.path.realpath(__file__))
from time import time
from rdkit import DataStructs
import ast
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import pickle
import time as time_mod

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False
            

def first_occurrences(data, start, end):
    """Indices in [start, end) keeping only the FIRST occurrence of each target.
    simple_200 holds 200 entries but only 135 unique molecules; with worker threads
    pulling from a shared queue, two copies of one molecule were worked on at the
    same time (2026-10-03: 8.8-14.3% of Easy LLM calls), racing on the same
    result/checkpoint files. Each unique molecule is now processed once."""
    seen, out = set(), []
    for idx in range(start, end):
        if data[idx] not in seen:
            seen.add(data[idx])
            out.append(idx)
    return out


def count_unfinished(data, indices, output_dir, seed):
    """Attempts among `indices` without a .done marker. `seed` may be one seed or a list of them
    (2026-11-20, --repeats): with N repeats a molecule counts once per unfinished repeat, so the
    `min_remaining` tail stop measures attempts left, which is what the workers actually have to do."""
    seeds = seed if isinstance(seed, (list, tuple, set)) else [seed]
    n = 0
    for idx in indices:
        for s in seeds:
            suffix = ("planning_" + data[idx] + "_" + str(s)).replace("/", "")
            if not os.path.exists(os.path.join(output_dir, f"results_{suffix}.done")):
                n += 1
    return n


def tail_stop(reason):
    """Config `min_remaining` (2026-10-03): stop a run once too few molecules remain
    to use the GPU sensibly. Creates jobs/STOP_<PBS job name> so the watchdog /
    end-of-chunk resubmission exits immediately instead of queueing empty chunks.
    In-progress molecules keep their checkpoints: delete the STOP file to resume."""
    name = os.environ.get("PBS_JOBNAME")
    if name:
        # Configurable so this file carries no site-specific path. LLM_SYN_JOBS_DIR overrides it.
        jobs_dir = os.environ.get("LLM_SYN_JOBS_DIR") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "jobs")
        os.makedirs(jobs_dir, exist_ok=True)
        stop = os.path.join(jobs_dir, f"STOP_{name}")
        with open(stop, "w") as f:
            f.write(f"tail-stop: {reason}\n")
        print(f"[tail-stop] {reason} -> wrote {stop}", flush=True)
    else:
        print(f"[tail-stop] {reason}", flush=True)


def run_threaded(optimizers, indices, handle):
    """Run handle(optimizer, idx, prefix) over `indices` with one thread per
    optimizer. Dynamic queue: an idle thread takes the next molecule, so a slow
    molecule never blocks a fixed block behind it. Each optimizer is used by
    exactly one thread. The first exception stops new work being handed out and
    is re-raised once running molecules finish."""
    import threading, queue
    todo = queue.Queue()
    for idx in indices:
        todo.put(idx)
    stop = threading.Event()
    errors = []

    def worker(k):
        while not stop.is_set():
            try:
                idx = todo.get_nowait()
            except queue.Empty:
                return
            try:
                handle(optimizers[k], idx, f"[Worker {k}] ")
            except BaseException as e:  # dead server etc.
                errors.append((idx, e))
                stop.set()
                return

    threads = [threading.Thread(target=worker, args=(k,), name=f"worker{k}")
               for k in range(len(optimizers))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        idx, e = errors[0]
        raise RuntimeError(f'worker failed on molecule idx={idx}: {type(e).__name__} {e}') from e


def main():
    start_time = time() 
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default=None)
    parser.add_argument('--dataset_name', default='pistachio_reachable')
    parser.add_argument('--config_default', default='config_local.yaml')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--max_oracle_calls', type=int, default=100)
    parser.add_argument('--freq_log', type=int, default=5)
    parser.add_argument('--seed', type=int, nargs="+", default=[0])
    # 2026-11-20 (user: "can we do 4x6 in one run so they essentially get 4 tries"): N independent
    # attempts per molecule, ALL IN ONE WAVE. --seed also gives independent attempts, but the seed loop
    # wraps the whole worker pool, so 4 seeds over 6 molecules is 4 sequential waves using 6 of 28
    # threads; --repeats queues every (molecule, attempt) pair together, so 6 molecules x 4 fills 24
    # threads at once. The attempt index goes into the result filename exactly where the seed does, so
    # each attempt has its own .yaml/.done/.ckpt and resumes independently. Duplicating rows in the
    # dataset file cannot do this: first_occurrences() de-duplicates targets on purpose, because two
    # threads on one molecule race on the same files (2026-10-03).
    parser.add_argument('--repeats', type=int, default=1,
                        help="independent attempts per molecule, run in one wave (default 1)")

    parser.add_argument('--log_results', action='store_true')
    parser.add_argument('--n_mols', type=int, default=None)
    parser.add_argument('--start_mol', type=int, default=0)
    parser.add_argument('--port', type=int, default=None,
                        help='Override LLM server port in config')
    parser.add_argument('--workers', type=int, default=1,
                        help='Molecules processed concurrently as THREADS in this one '
                             'process, sharing a single copy of the inventory/templates/'
                             'fingerprints (~24 GB). Replaces launching N separate '
                             'main.py processes, which held N copies (250 GB at N=10).')
    args = parser.parse_args()



    args.method = args.method.lower()


    
    print(args.method)
    if args.method == 'planning':
        from run import planning_Optimizer as Optimizer 

    else:
        raise ValueError("Unrecognized method name.")

    train_data = 'dataset/routes_train.pkl'
    val_data = 'dataset/routes_val.pkl'
    test_data = 'dataset/routes_test.pkl'
    test_hard_data = 'dataset/routes_possible_test_hard.pkl'
    train_routes = pickle.load(open(train_data, 'rb'))
    val_routes = pickle.load(open(val_data, 'rb'))
    test_routes = pickle.load(open(test_data, 'rb'))
    test_hard_routes = pickle.load(open(test_hard_data, 'rb'))

    total_routes = train_routes + val_routes
    target_list = []
    route_list = []
    for route in total_routes:
        target_list.append(route[0].split('>>')[0])
        route_list.append(route)
    
    test_list = []
    test_route_list = []
    for route in test_routes:
        test_list.append(route[0].split('>>')[0])
        test_route_list.append(route)
    
    test_hard_list = []
    test_hard_route_list = []
    for route in test_hard_routes:
        test_hard_list.append(route[0].split('>>')[0])
        test_hard_route_list.append(route)

    # Ground-truth precursors (parallel to `data`), used by Run S v2 to weight
    # skills by how close a proposed route's first disconnection is to the known
    # precursor. Only the pistachio_* tuple datasets carry this; others stay None.
    gt_data = None

    if args.dataset_name == 'USPTO-easy':
        with open('dataset/USPTO_easy.pkl', 'rb') as file:
            test_mol = pickle.load(file)
        data = test_mol
    elif args.dataset_name == 'USPTO-190':
        test_hard_list = []
        test_hard_route_list = []
        for route in test_hard_routes:
            test_hard_list.append(route[0].split('>>')[0])
            test_hard_route_list.append(route)
        data = test_hard_list
    elif args.dataset_name == 'pistachio_reachable':
        file_path = "dataset/pistachio_reachable_targets.txt"
        data = []
        gt_data = []

        with open(file_path, "r") as file:
            for line in file:
                #
                line = line.strip()
                if line:  #
                    try:
                        #
                        molecule_tuple = eval(line)
                        #
                        first_molecule = molecule_tuple[0]
                        #
                        data.append(first_molecule)
                        gt_data.append(molecule_tuple[1] if len(molecule_tuple) > 1 else None)
                    except Exception as e:
                        print(f"{line}, error info: {e}")
    
    elif args.dataset_name == 'test_1mol':
        file_path = "dataset/test_1mol.txt"
        data = []
        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()
                if line:
                    try:
                        molecule_tuple = eval(line)
                        first_molecule = molecule_tuple[0]
                        data.append(first_molecule)
                    except Exception as e:
                        print(f"{line}, error info: {e}")

    elif args.dataset_name == 'simple_200':
        with open('dataset/simple_200.pkl', 'rb') as file:
            data = pickle.load(file)
    elif args.dataset_name == 'run_C_extremes_20':
        with open('dataset/run_C_extremes_20.pkl', 'rb') as file:
            data = pickle.load(file)
    elif args.dataset_name in ('pistachio_hard', 'pistachio_hard_losses10',
                               'pistachio_hard_oracle22', 'pistachio_hard_upside12',
                               'ablation_30', 'pistachio_hard_tfix28',
                               'pistachio_hard_capsskills26', 'pistachio_hard_strat28',
                               'pistachio_hard_slow30', 'pistachio_hard_slow15',
                               'pistachio_hard_cfail15', 'pistachio_hard_skill6'):
        file_path = {
            'pistachio_hard_losses10': "dataset/pistachio_hard_losses10.txt",
            'pistachio_hard_oracle22': "dataset/pistachio_hard_oracle22.txt",
            'pistachio_hard_upside12': "dataset/pistachio_hard_upside12.txt",
            # 30-molecule prompt-ablation subset: drawn from molecules baseline A
            # has already completed, preserving A's 70/30 pass/fail mix so each
            # arm is directly comparable to existing A results without a re-run.
            'ablation_30': "dataset/ablation_30.txt",
            # 2026-10-04 template-choice test: the 22 pistachio_hard molecules baseline
            # A_b100 failed + 6 random baseline passes as controls (mapping to the
            # original indices in dataset/pistachio_hard_tfix28.json).
            'pistachio_hard_tfix28': "dataset/pistachio_hard_tfix28.txt",
            # 2026-10-12: 16 AOT v3 budget failures + the 10 solved molecules that used most LLM calls
            # (original indices in dataset/pistachio_hard_capsskills26.json).
            'pistachio_hard_capsskills26': "dataset/pistachio_hard_capsskills26.txt",
            # 2026-10-18 advisor-fix test set: 7 easy / 7 medium / 7 hard / 7 unsolved by AOT v3's result
            # (strata and original indices in dataset/pistachio_hard_strat28.json; built by analysis/build_strat28.py).
            'pistachio_hard_strat28': "dataset/pistachio_hard_strat28.txt",
            # 2026-11-14 (user: "the 30 longest time take molecule" in the recent 100-molecule control): the 30
            # molecules that cost results/run_A_27B/pistachio_hard_aot3 the most LLM time. Selected on the CONTROL's
            # cost only, never on a tool arm, so a cutselR-based arm scored against hard100_cutselR on this set is
            # not biased by the selection (original indices + per-molecule control hours and cutselR's result in
            # dataset/pistachio_hard_slow30.json).
            'pistachio_hard_slow30': "dataset/pistachio_hard_slow30.txt",
            # 2026-11-14 (user: "do only the last 15, I just want to know if it helps"): the FIRST 15 of slow30,
            # i.e. the 15 slowest of all in the control. At the gate of 75 the user chose, the LAST 15 would have
            # fired on only 2 molecules, so the slowest 15 is the subset that can answer the question
            # (indices + cutselR's result + the coverage warning in dataset/pistachio_hard_slow15.json).
            'pistachio_hard_slow15': "dataset/pistachio_hard_slow15.txt",
            # 2026-11-19 (user: "Add them to the 15 remove two worst ones and queue again"): slow15 with the two
            # cheapest unanimously-solved molecules dropped and cutselR's two BUDGET failures from OUTSIDE slow15
            # added. slow15 was picked on the CONTROL's wall time, so it held only 2 of cutselR's 4 failures and an
            # arm scored against hard100_cutselR could win at most +2 there; here the ceiling is +4. Still n=15, so
            # this is a screen: 4-0 is p=0.0625. Provenance in dataset/pistachio_hard_cfail15.json.
            'pistachio_hard_cfail15': "dataset/pistachio_hard_cfail15.txt",
            # 2026-11-20 (user: "submit it with the 4x on the 6 molecules in same run"): the 6 targets the
            # skill library in skill_library_pg.json was derived from -- ctoolselfS's 4 misses plus the 2 it
            # solved once and lost in its replicate. Run with --repeats 4: each target gets four independent
            # attempts in one wave, so the measurement is a SOLVE RATE out of 4 rather than a coin-flip.
            # Two of the six have never been solved by any arm, so the ceiling is 4 of 6. Provenance and the
            # per-target historical rates are in dataset/pistachio_hard_skill6.json.
            'pistachio_hard_skill6': "dataset/pistachio_hard_skill6.txt",
        }.get(args.dataset_name, "dataset/pistachio_hard_targets.txt")
        data = []
        gt_data = []
        with open(file_path, "r") as file:
            for line in file:
                #
                line = line.strip()
                if line:  #
                    try:
                        molecule_tuple = eval(line)
                        first_molecule = molecule_tuple[0]
                        data.append(first_molecule)
                        gt_data.append(molecule_tuple[1] if len(molecule_tuple) > 1 else None)
                    except Exception as e:
                        print(f"{line}, error info: {e}")
    elif args.dataset_name == 'case_study':
        file_path = "dataset/case_study.txt"
        data = []
        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()
                if line:  # 
                    try:
                        molecule_tuple = eval(line)
                        first_molecule = molecule_tuple[0]
                        data.append(first_molecule)
                    except Exception as e:
                        print(f"{line}, error info: {e}")
    # Datasets without ground-truth precursors get a None-filled parallel list so
    # the optimize() call below can index it uniformly.
    if gt_data is None or len(gt_data) != len(data):
        gt_data = [None] * len(data)

    similarity_metric = DataStructs.BulkTanimotoSimilarity # BulkDiceSimilarity or BulkTanimotoSimilarity
    similarity_label = 'Tanimoto'
    getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=False)
    getfp_label = 'Morgan2noFeat'

    all_fps = []
    for smi in target_list:
        all_fps.append(getfp(smi))
    

    if args.output_dir is None:
        args.output_dir = os.path.join(args.dataset_name, 'results')
    os.makedirs(args.output_dir, exist_ok=True)


    config_default = yaml.safe_load(open(args.config_default))

    # Config `aot.split_pool` (2026-11-11, off by default). Validated HERE, before the ~25 GB data load, so a bad
    # or unsafe setting stops the job in seconds (aot_search.split_pool_settings says what it checks).
    _sp_c = None
    if (config_default.get('aot') or {}).get('split_pool'):
        from aot_search import split_pool_settings
        _sp_c = split_pool_settings(config_default.get('aot'), config_default)
        if not _sp_c['enabled']:
            _sp_c = None
        elif args.method != 'planning' or args.workers <= 1 or len(args.seed) != 1:
            raise SystemExit("aot.split_pool needs --method planning, --workers > 1 and exactly one --seed "
                             "(the pool is forked once, before the worker threads)")
        else:
            # The pool can only be forked while this process has ONE thread. tqdm's progress-bar monitor is a
            # thread, so it is switched off for the data load below (it only refreshes stalled bars).
            import tqdm.std as _tqdm_std
            _tqdm_std.tqdm.monitor_interval = 0

    # Tail stop (config `min_remaining`): don't spend a whole GPU on a handful of
    # molecules. Checked before the ~25 GB data load and again every minute.
    min_remaining = int(config_default.get('min_remaining', 0) or 0)
    _start = args.start_mol
    _end = _start + args.n_mols if args.n_mols is not None else len(data)
    if min_remaining and args.method == 'planning':
        _rp0 = max(int(getattr(args, "repeats", 1) or 1), 1)
        _sd0 = [f"{args.seed[0]}r{r}" for r in range(_rp0)] if _rp0 > 1 else args.seed[0]
        _left = count_unfinished(data, first_occurrences(data, _start, _end), args.output_dir, _sd0)
        if _left < min_remaining:
            tail_stop(f"only {_left} molecules unfinished < min_remaining {min_remaining}; not starting")
            return

    if args.port is not None:
        config_default['llm']['server_url'] = f'http://127.0.0.1:{args.port}/v1'

    # Load shared data once to save memory in parallel runs
    from optimizer import preprocess_reaction_dict
    import json
    args.template_path = 'dataset/idx2template_retro.json'
    args.inventory_path = 'dataset/inventory.pkl'
    args.rule_based_set_path = 'dataset/data_processed.csv'
    
    # Load templates
    with open(args.template_path, "r") as f:
        original_template_dict = json.load(f)
    template_dict = preprocess_reaction_dict(original_template_dict)
    
    # Load inventory
    with open(args.inventory_path, 'rb') as file:
        inventory = pickle.load(file)
    # Pickle was built with older syntheseus (_mol_set); current version needs _smiles_set
    if hasattr(inventory, '_mol_set') and not hasattr(inventory, '_smiles_set'):
        inventory._smiles_set = {mol.smiles for mol in inventory._mol_set}
    
    # Get reaction fps
    from rdkit.Chem import rdChemReactions
    reaction_list = list(original_template_dict.values())
    getreactionfp = lambda smart_reaction: rdChemReactions.CreateDifferenceFingerprintForReaction(rdChemReactions.ReactionFromSmarts(smart_reaction))
    all_reaction_fps = []
    for reaction in reaction_list:
        all_reaction_fps.append(getreactionfp(reaction))
    
    # Load rule-based set
    from sim_based_rag import get_data_df, split_data_df
    from tqdm import tqdm
    data_df = get_data_df(args.rule_based_set_path)
    split_data_df(data_df)
    getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=False)
    all_fps_df = []
    for smi in tqdm(data_df['prod_smiles']):
        all_fps_df.append(getfp(smi))
    data_df['prod_fp'] = all_fps_df
    datasub = data_df.loc[data_df['dataset'] == 'train']
    fps = list(datasub['prod_fp'])
    print('Size of knowledge base: {}'.format(len(fps)))
    
    template_to_key = {v: k for k, v in original_template_dict.items()}
    
    shared_data = {
        'original_template_dict': original_template_dict,
        'template_dict': template_dict,
        'inventory': inventory,
        'reaction_list': reaction_list,
        'all_reaction_fps': all_reaction_fps,
        'datasub': datasub,
        'template_to_key': template_to_key,
    }

    if args.method == "planning":
        # The TensorBoard writer starts a background thread, so it is opened only AFTER the split pool has been
        # forked (below): with it open first, split_pool.start() refused to fork and every pooled run would have
        # silently run in-process (review, 12 Nov). `process` reads tb_writer when it is called, not now.
        tb_writer = None

        def _open_tb(w):
            if w is not None or not _TB_AVAILABLE:
                return w
            tb_dir = os.path.join(args.output_dir, 'tensorboard')
            print(f'TensorBoard logging → {tb_dir}')
            return SummaryWriter(log_dir=tb_dir)

        start = args.start_mol
        end = start + args.n_mols if args.n_mols is not None else len(data)
        n_mols = end - start
        
        # Compute worker ID for logging (if --start_mol and --n_mols are set)
        worker_id = None
        if args.n_mols is not None and args.n_mols > 0:
            worker_id = args.start_mol // args.n_mols

        # `search: aot` selects the AOT* AND-OR tree search (aot_search.py, a new
        # addition); absent = the original LLM-Syn-Planner GA, unchanged.
        if config_default.get('search') == 'aot':
            from aot_search import AOTOptimizer as Optimizer
        else:
            from run import planning_Optimizer as Optimizer

        def process(optimizer, idx, seed, prefix):
            target = data[idx]
            suffix = ("planning_" + target + "_" + str(seed)).replace("/", "")
            done_file = os.path.join(args.output_dir, f'results_{suffix}.done')
            result_file = os.path.join(args.output_dir, f'results_{suffix}.yaml')
            # Skip only if molecule is marked done (PASS/BUDGET/EARLY_TERM).
            # A stale .yaml without .done = walltime-cut or crash: re-attempt.
            if os.path.exists(done_file):
                print(f'{prefix}[{idx+1}/{n_mols}] Skipping (done): {target}')
                return
            if os.path.exists(result_file):
                print(f'{prefix}[{idx+1}/{n_mols}] Re-attempting (no .done marker — prior run cut off; resumes from .ckpt.pkl if present): {target}')
            print(f'{prefix}[{idx+1}/{n_mols}] Searching routes for: {target}')
            optimizer.optimize(target, route_list, all_fps,
                               config=config_default, seed=seed,
                               tb_writer=tb_writer, target_idx=idx,
                               ground_truth=gt_data[idx])

        for seed in args.seed:
            if args.workers <= 1:
                tb_writer = _open_tb(tb_writer)
                optimizer = Optimizer(args=args, config=config_default, shared_data=shared_data)
                worker_prefix = f"[Worker {worker_id}] " if worker_id is not None else ""
                for idx in first_occurrences(data, start, end):
                    process(optimizer, idx, seed, worker_prefix)
                continue

            from utils import is_reaction_in_dict
            # RDKit computes some query-mol properties lazily on first match; do
            # that once here, single-threaded, before threads share the templates.
            is_reaction_in_dict('CCO>>CC.O', template_dict)
            # Build every optimizer up front in the main thread (model/scorer loads
            # are not guaranteed thread-safe); each thread then owns one.
            optimizers = [Optimizer(args=args, config=config_default, shared_data=shared_data)
                          for _ in range(args.workers)]
            # Config `aot.split_pool` (2026-11-11, off by default): fork the split-search worker processes HERE.
            # This is the last single-threaded moment -- the data is loaded, the optimizers exist, and no worker,
            # monitor or TensorBoard thread has started yet, so the children inherit the ~25 GB read-only through
            # copy-on-write and cannot inherit a lock another thread holds. split_pool.start() refuses to fork
            # if that is ever no longer true, and then the JOB STOPS: an arm configured as pooled must never run
            # unpooled under that label (review, 12 Nov: that is exactly what the TensorBoard thread caused).
            if _sp_c:
                import split_pool
                if not split_pool.start(optimizers[0], _sp_c['processes'], _sp_c['timeout_s'], _sp_c['wait_s']):
                    raise SystemExit("aot.split_pool is enabled but the pool did not start (see the "
                                     "[split_pool] line above); stopping rather than running unpooled")
            tb_writer = _open_tb(tb_writer)   # starts a thread: only after the fork
            if min_remaining:
                import threading as _th

                def _tail_monitor(indices=first_occurrences(data, start, end)):
                    while True:
                        time_mod.sleep(60)
                        _rp = max(int(getattr(args, "repeats", 1) or 1), 1)
                        _sds = [f"{seed}r{r}" for r in range(_rp)] if _rp > 1 else seed
                        left = count_unfinished(data, indices, args.output_dir, _sds)
                        if left < min_remaining:
                            tail_stop(f"{left} molecules unfinished < min_remaining {min_remaining}; "
                                      f"stopping mid-chunk (in-progress molecules keep checkpoints)")
                            os._exit(0)   # checkpoints are written atomically (tmp + replace)
                _th.Thread(target=_tail_monitor, daemon=True, name="tail-monitor").start()
            try:
                # --repeats: one task per (molecule, attempt) so every attempt is in this single wave
                _reps = max(int(getattr(args, "repeats", 1) or 1), 1)
                _idxs = first_occurrences(data, start, end)
                if _reps > 1:
                    _tasks = [(idx, f"{seed}r{r}") for idx in _idxs for r in range(_reps)]
                    run_threaded(optimizers, _tasks,
                                 lambda opt, t, prefix: process(opt, t[0], t[1], prefix))
                else:
                    run_threaded(optimizers, _idxs,
                                 lambda opt, idx, prefix: process(opt, idx, seed, prefix))
            finally:
                if _sp_c:
                    import split_pool
                    print(f"[split_pool] {split_pool.STATS}")
                    split_pool.stop()

        if tb_writer is not None:
            tb_writer.close()
    

    end_time = time()
    hours = (end_time - start_time) / 3600.0
    print('---- The whole process takes %.2f hours ----' % (hours))



if __name__ == "__main__":
    main()