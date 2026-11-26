import os
import sys
import yaml
import random
import torch
import numpy as np
from rdkit import Chem
import copy
import tdc
import json
import re
import heapq
import fcntl
import threading
import hashlib
import time as _time
from datetime import datetime as _datetime
from collections import Counter as _Counter
from rdkit import Chem
from syntheseus import Molecule
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions, DataStructs
from itertools import permutations
from utils import *
from scscore.scscore.standalone_model_numpy import *
import pickle
import json
from rdchiral.main import rdchiralRun
from rdchiral.initialization import rdchiralReactants, rdchiralReaction
from sim_based_rag import get_data_df, split_data_df, do_one
from tqdm import tqdm
from llm_backend import get_client, get_model_name

# main.py --workers N runs N molecules as threads in ONE process (one copy of
# the ~24 GB shared data instead of one per process). Every shared append-mode
# log goes through this lock so concurrent lines can't interleave.
_APPEND_LOCK = threading.Lock()


def split_think(text, finish=None):
    """Config `llm.strip_think` (2026-10-16, Intern-S2 as the planner): a thinking model served
    without a reasoning parser returns "<reasoning></think><answer>" as its content, and the
    reasoning holds draft <ROUTE> blocks. Returns (answer, reasoning): the answer is the text after
    the LAST "</think>". No closing tag and cut at max_tokens -> ("", text): the model was still
    thinking, so there is no answer (as an advisor note cut at the cap gives no advice). No closing
    tag otherwise -> (text, "") unchanged."""
    if text is None:
        return None, ""
    if "</think>" in text:
        head, _, tail = text.rpartition("</think>")
        return tail.strip(), head.replace("<think>", "", 1).strip()
    if finish == "length":
        return "", text
    return text, ""


# Injected on the forced-answer turn of the agentic (Run G) tool loop: the model has used
# its tool budget but tends to keep working step-by-step rather than emit a route, so we
# explicitly tell it to stop calling tools and serialize the complete route NOW, in the
# exact Python-literal format the downstream <ROUTE> parser expects.
_AGENTIC_FINALIZE_MSG = (
    "You have reached your tool-call budget. Do NOT call any more tools and do NOT write any "
    "tool-call syntax. Write your FINAL answer directly as text, using the best route you have "
    "built so far (a PARTIAL route is fine — it will be refined later), in EXACTLY this format:\n"
    "<PLAN> your plan </PLAN>\n"
    "<EXPLANATION> brief explanation </EXPLANATION>\n"
    "<ROUTE>\n[ {'Molecule set': \"[...]\", 'Rational': '...', 'Product': \"[...]\", "
    "'Reaction': \"[Product>>Reactant1.Reactant2]\", 'Reactants': \"[...]\", "
    "'Updated molecule set': \"[...]\"}, ... ]\n</ROUTE>\n"
    "The <ROUTE> must be a valid Python list of step dicts with at least 2 steps. "
    "Begin your reply with <PLAN> now."
)

class Route:
    def __init__(self, target):
        self.target = target
        self.raw_output = raw_output

    def get_score():
        sc_score = []




def preprocess_reaction_dict(reaction_dict):
    """Preprocess reaction_dict to compile SMARTS into RDKit Mol objects."""
    preprocessed_dict = {}
    for key, smarts in reaction_dict.items():
        try:
            products, reactants = smarts.split(">>")
            reactant_mols = [Chem.MolFromSmarts(r) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmarts(p) for p in products.split(".")]
            preprocessed_dict[key] = (reactant_mols, product_mols)
        except Exception as e:
            print(f"Error preprocessing SMARTS {smarts}: {e}")
    return preprocessed_dict

#Return SC score and store routes
class Oracle:
    def __init__(self, args=None, route_buffer=None):
        self.name = None
        self.evaluator = None
        self.task_label = None
        if args is None:
            self.max_oracle_calls = 10000
            self.freq_log = 100
        else:
            self.args = args
            self.max_oracle_calls = args.max_oracle_calls
            self.freq_log = args.freq_log
        self.route_buffer = {} if route_buffer is None else route_buffer
        self.reaction_cache = dict() #mol_smiles: [reaction]

        self.last_log = 0
        self.sa_scorer = tdc.Oracle(name='SA')
        self.sc_Oracle = SCScorer()
        self.sc_Oracle.restore()

    def store_cache(self, mol_smiles, reaction):
        if mol_smiles in self.reaction_cache:
            if reaction not in self.reaction_cache[mol_smiles]:
                self.reaction_cache[mol_smiles].append(reaction)
        else:

            self.reaction_cache[mol_smiles] = [reaction]

    def get_oracle_score(self, mol_smiles):
        smi, SC_score = self.sc_Oracle.get_score_from_smi(mol_smiles)
        sa_score = self.sa_scorer(mol_smiles)
        length = len(self.route_buffer)
        if length <= 150:
            overall_score = SC_score
        elif length > 150 and length < 220:
            alpha = (length/self.max_oracle_calls)
            overall_score = (1 - alpha) * SC_score + 0.5 * alpha * sa_score            
        else:
            overall_score = 0.5 * sa_score 
        #overall_score = SC_score + 0.5 * sa_score
        return overall_score
    @property
    def budget(self):
        return self.max_oracle_calls
    
    def reward(self, inventory, updated_molecule_set:list, visited_molecules, dead_molecules):
        score_list = []
        for smi in updated_molecule_set:
            if smi in dead_molecules:
                if dead_molecules[smi] >= 1:
                    print('dead molecules!')
                    score_list.append(100)
                    continue
            try:
                signal = inventory.is_purchasable(Molecule(smi))
                if not signal:
                    score = self.get_oracle_score(smi)
                    if smi in visited_molecules:
                        print(f"Visted times: {visited_molecules[smi]}")
                        if visited_molecules[smi] > 15:
                            score = (visited_molecules[smi]/15) * score
                            print(f"Visted times adjust score")
                    score_list.append(score)
            except Exception as e:
                print(f"Error: {e}")
                score_list.append(5)
        if len(score_list) != 0:
            score_max = np.max(score_list)
        else:
            score_max = 0
        if len(score_list) != 0:
            score_mean = sum(score_list) / len(score_list) 
        else:
            score_mean = 0
        combined_score = score_mean + sum(score_list)
        final_score = - combined_score
        return final_score
    
    def evaluate(self, inventory, route_evaluation, visited_molecules, dead_molecules):
        #print(route_evaluation)
        for idx, step in enumerate(route_evaluation):
            #print(step)
            if step[1] == False:
                score = self.reward(inventory, step[2]['molecule_set'], visited_molecules, dead_molecules)
                return score
            elif step[1] == True:
                continue
        
        #last step
        print(route_evaluation[-1])
        if route_evaluation[-1][2]['check_availability'] == True and len(route_evaluation[-1][2]['unavailable_mol_id']) == 0:
            score = 0
            return score
        else:
            score = self.reward(inventory, route_evaluation[-1][2]['updated_molecule_set'], visited_molecules, dead_molecules)
            return score

    def sort_buffer(self):
        self.route_buffer = dict(sorted(self.route_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    def save_result(self, suffix=None):
        
        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            suffix = suffix.replace("/", "")
            output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + '.yaml')

        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.route_buffer, f, sort_keys=False)

    def log_intermediate(self, finish=False):
        if finish:
            n_calls = self.max_oracle_calls
            self.save_result(self.task_label)

    def __len__(self):
        return len(self.route_buffer) 

    def score_route(self, inventory, route_evaluation, visited_molecules, dead_molecules):
        """
        Function to score one molecule

        Argguments:
            smi: One SMILES string represnets a moelcule.

        Return:
            score: a float represents the property of the molecule.
        """
        if len(self.route_buffer) > self.max_oracle_calls:
            return -15
        if route_evaluation is None:
            return -15
        dict_key = json.dumps(route_evaluation)
        if dict_key in self.route_buffer:
            pass
        else:
            self.route_buffer[dict_key] = [float(self.evaluate(inventory, route_evaluation, visited_molecules, dead_molecules)), len(self.route_buffer)+1]
        return self.route_buffer[dict_key][0]
    
    def __call__(self, inventory, route_evaluation, visited_molecules, dead_molecules):
        """
        Score
        """
        score_list = self.score_route(inventory, route_evaluation, visited_molecules, dead_molecules)
        if len(self.route_buffer) % self.freq_log == 0 and len(self.route_buffer) > self.last_log:
            self.sort_buffer()
            self.last_log = len(self.route_buffer)
            self.save_result(self.task_label)
        return score_list

    @property
    def finish(self):
        return len(self.route_buffer) >= self.max_oracle_calls


class BaseOptimizer:

    def __init__(self, args=None, config=None, shared_data=None):
        self.model_name = "Default"
        self.args = args

        # Load config and init LLM client
        # Use pre-loaded config if provided (preserves --port override from main.py),
        # otherwise fall back to reading the file.
        if config is not None:
            self.llm_config = config
        else:
            self.llm_config = yaml.safe_load(open(args.config_default)) if args else {}
        self.llm_client = get_client(self.llm_config)
        self.llm_model_name = get_model_name(self.llm_config)
        self._consecutive_conn_failures = 0
        self._max_consecutive_conn_failures = 20  # bail after 20 consecutive connection errors

        self.oracle = Oracle(args=self.args)

        if shared_data is not None:
            # Use pre-loaded shared data
            self.original_template_dict = shared_data['original_template_dict']
            self.template_dict = shared_data['template_dict']
            self.inventory = shared_data['inventory']
            self.reaction_list = shared_data['reaction_list']
            self.all_reaction_fps = shared_data['all_reaction_fps']
            self.datasub = shared_data['datasub']
            self.template_to_key = shared_data['template_to_key']
        else:
            # Load data (legacy mode)
            args.template_path = 'dataset/idx2template_retro.json'
            args.inventory_path = 'dataset/inventory.pkl'
            args.rule_based_set_path = 'dataset/data_processed.csv'
            self.original_template_dict = self.load_template(args.template_path)
            self.template_dict =  preprocess_reaction_dict(self.original_template_dict)
            self.inventory = self.load_inventory(args.inventory_path)
            self.reaction_list, self.all_reaction_fps = self.get_reaction_fps(self.original_template_dict)
            self.datasub = self.load_rule_based_set(args.rule_based_set_path)
            self.template_to_key = {v: k for k, v in self.original_template_dict.items()}

        # Per-instance state
        self.explored_reaction = set()
        self.visited_molecules = dict() #smiles: visit number
        self.dead_molecules = dict()
        self.jx_cache = {}

        # Optional retrosynthesis strategy guide (injected into init + modification prompts
        # when strategy_guide_path is set in config). Mirrors the TriFusion convention.
        self._strategy_context = ''
        guide_path = self.llm_config.get('strategy_guide_path', '')
        if guide_path:
            try:
                with open(guide_path) as _gf:
                    self._strategy_context = _gf.read()
                print(f'[Optimizer] Loaded strategy guide ({len(self._strategy_context)} chars): {guide_path}')
            except Exception as _ge:
                print(f'[Optimizer] WARNING: failed to load strategy guide {guide_path}: {_ge}')

        # Optional evolving skill-structured strategy memory (Run S). Replaces the
        # flat guide above with a categorized, statistic-bearing skill store that is
        # bandit-sampled per molecule and evolved after each one. See skill_memory.py.
        self._skill_cfg = self.llm_config.get('skill_memory', {}) or {}
        self._skill_memory_enabled = bool(self._skill_cfg.get('enabled', False))
        self._skill_memory_path = self._skill_cfg.get('path', '')
        self._skill_memory = None
        self._injected_skill_ids = []
        self._injected_skill_block = None  # cached per molecule
        if self._skill_memory_enabled:
            import skill_memory as _sm
            import skill_signature as _sig
            self._sm = _sm
            self._sig = _sig
            self._skill_memory = _sm.load_memory(self._skill_memory_path)
            print(f'[Optimizer] Skill memory enabled ({len(self._skill_memory.get("skills", []))} '
                  f'skills): {self._skill_memory_path}')


    def load_template(self, template_path):
        with open(template_path, "r") as f:
            template_dict = json.load(f)
        #preprocessed_dict = preprocess_reaction_dict(template_dict)
        return template_dict

    
    def load_inventory(self, inventory_path):
        with open(inventory_path, 'rb') as file:
            inventory = pickle.load(file)
        # Pickle was built with older syntheseus (_mol_set); current version needs _smiles_set
        if hasattr(inventory, '_mol_set') and not hasattr(inventory, '_smiles_set'):
            inventory._smiles_set = {mol.smiles for mol in inventory._mol_set}
        return inventory
    
    def load_rule_based_set(self, rule_based_set_path):
        data = get_data_df(rule_based_set_path)
        split_data_df(data)
        similarity_metric = DataStructs.BulkTanimotoSimilarity # BulkDiceSimilarity or BulkTanimotoSimilarity

        getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=False)

        dataset = 'val'

        all_fps = []
        for smi in tqdm(data['prod_smiles']):
            all_fps.append(getfp(smi))
        data['prod_fp'] = all_fps

        datasub = data.loc[data['dataset'] == 'train']
        fps = list(datasub['prod_fp'])
        print('Size of knowledge base: {}'.format(len(fps)))
        
        return datasub

    def update_visited_molecules(self, updated_molecule_set):
        for smi in updated_molecule_set:
            if smi in self.visited_molecules:
                self.visited_molecules[smi] += 1
            else:
                self.visited_molecules[smi] = 1

    def update_dead_molecules(self, dead_molecule):
        smi = dead_molecule
        assert type(smi) == str
        print('update dead molecules!')
        print(smi)
        if smi in self.dead_molecules:
            self.dead_molecules[smi] += 1
        else:
            self.dead_molecules[smi] = 1
                
    def get_reaction_fps(self, template_dict):
        reaction_list = list(template_dict.values())
        getreactionfp = lambda smart_reaction: rdChemReactions.CreateDifferenceFingerprintForReaction(rdChemReactions.ReactionFromSmarts(smart_reaction))
        all_reaction_fps = []
        for reaction in reaction_list:
            all_reaction_fps.append(getreactionfp(reaction))
        
        return reaction_list, all_reaction_fps
    
    def convert_to_fingerprint(self, mols_list):
        fgp_list = []
        getfp = lambda mol: AllChem.GetMorganFingerprint(mol, 2, useFeatures=False)
        for mol in mols_list:
            fgp_list.append(getfp(mol))
        
        return fgp_list

    def rule_based_search(self, product_smiles, reaction_smiles):
        template_list, self.jx_cache = do_one(product_smiles, self.datasub, self.jx_cache)

        if template_list == []:
            print('sim cannot found')
            return False, None, reaction_smiles
        else:
            templates = [t[0] for t in template_list]
            scores = [t[1] for t in template_list]
            weights = np.array(scores) 
            probabilities = weights / weights.sum()  # Normalize to get probabilities
            sampled_index = np.random.choice(len(templates), p=probabilities, size=len(templates), replace=False)
            sorted_templates = [templates[i] for i in sampled_index]
            for template in sorted_templates:
                raw_template = template[1:].replace(')>>', '>>')
                if (product_smiles, raw_template) in self.explored_reaction:
                    continue
                else:
                    key = '99999999999'
                    print('sim based reaction found')
                    return True, key, raw_template
            print('sim cannot found')
            return False, None, reaction_smiles
    def sanitize_smiles(self, smiles):
        """
        Check if a SMILES string is valid and return the sanitized molecule.

        Parameters:
            smiles (str): SMILES string.

        Returns:
            str or None: Sanitized SMILES string if valid, None otherwise.
        """
        if smiles == '':
            return None
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
            smi_canon = Chem.MolToSmiles(mol, canonical=True)
            return smi_canon
        except:
            return None

    def sanitize_reaction(self, reaction_smiles):
        """
        Process a reaction SMILES, removing invalid molecules in reactants and products.

        Parameters:
            reaction_smiles (str): Reaction SMILES in the format "reactants>>products".

        Returns:
            str: Sanitized reaction SMILES with only valid reactants and products.
        """
        try:
            reactants, products = reaction_smiles.split('>>')
        
            reactants_list = reactants.split('.')
            products_list = products.split('.')

            sanitized_reactants = [self.sanitize_smiles(smiles) for smiles in reactants_list]
            sanitized_products = [self.sanitize_smiles(smiles) for smiles in products_list]

            sanitized_reactants = [s for s in sanitized_reactants if s is not None]
            sanitized_products = [s for s in sanitized_products if s is not None]

            sanitized_reaction = ".".join(sanitized_reactants) + ">>" + ".".join(sanitized_products)

            return sanitized_reaction

        except Exception as e:
            print(f"Invalid reaction SMILES format: {reaction_smiles}. Error: {e}")
            return reaction_smiles

    def blurry_search(self, reaction_smiles, product_smiles, exploration_signal):
        similarity_metric = DataStructs.BulkTanimotoSimilarity
        if exploration_signal == True:
            reaction_number = 1000
        else:
            reaction_number = 100

        try:
            # remove invalid molecules in the reaction to perform similarity search
            sanitized_reaction = self.sanitize_reaction(reaction_smiles)
            #sanitized_reaction = reaction_smiles
            rxn_obj = smiles_to_reaction(sanitized_reaction)
            fp_re = rdChemReactions.CreateDifferenceFingerprintForReaction(rxn_obj)

            sims = similarity_metric(fp_re, self.all_reaction_fps)
            # heapq.nlargest to fetch top reaction_number indexes
            top_indices = heapq.nlargest(reaction_number, range(len(sims)), key=lambda i: sims[i])
            sorted_reaction_list = [self.reaction_list[i] for i in top_indices]
            
        except Exception as e:
            print(f"Error {e} getting reaction {reaction_smiles} for fingerprints!")
            #return False, None, reaction_smiles
            return self.rule_based_search(product_smiles, reaction_smiles)

        try:
            target_rd = rdchiralReactants(product_smiles)
        except Exception as e:
            print(f"Error {e} initializing rdchiralReactants for product {product_smiles}")
            return False, None, reaction_smiles

        for reaction_smarts in sorted_reaction_list:
            try:
                reaction_outputs = run_retro(target_rd, reaction_smarts)
                if len(reaction_outputs) > 1:
                    reaction_outputs = self.rank_reactants(reaction_outputs)
                if len(reaction_outputs) == 0:
                    continue
                reactants_generated = [reactant for reactant in reaction_outputs[0]]
                if reactants_generated == []:
                    continue
                elif len(reactants_generated) > 0:
                    key = self.template_to_key.get(reaction_smarts)
                    if (product_smiles, reaction_smarts) in self.explored_reaction:
                        print('redundant!')
                        continue
                    return True, key, reaction_smarts
            except Exception as e:
                print(f"Error {e} testing reaction {reaction_smarts} on product {product_smiles}")
                continue
        return self.rule_based_search(product_smiles, reaction_smiles)

    def _step1_substitute(self, reaction_smiles, product_smiles, model_reactants, exploration_signal, why):
        """check_route's step-1 substitution. Default (= upstream): blurry_search, i.e. the silent
        switch. AOT switch choice (`aot.switch_choice`, sets self._switch_choice_on): no silent
        switch. A template the model chose (self._switch_forced, same product) is used; otherwise
        the near-miss options are stored in self._switch_pending and the step is left invalid."""
        if not getattr(self, "_switch_choice_on", False):
            return self.blurry_search(reaction_smiles, product_smiles, exploration_signal)
        found = self.switch_options(reaction_smiles, product_smiles, model_reactants, exploration_signal,
                                    k=2 * int(getattr(self, "_switch_max_options", 3)))   # some fail the dry run
        self._switch_pending = dict(found, product=product_smiles, reason=why, reaction=reaction_smiles,
                                    reactants=list(model_reactants or []))
        return False, None, reaction_smiles

    def _switch_forced_for(self, product_smiles):
        """(True, key, template) of the model's switch-choice pick for this product, else None."""
        forced = getattr(self, "_switch_forced", None) if getattr(self, "_switch_choice_on", False) else None
        if forced and forced["product"] == product_smiles:
            return True, forced["key"], forced["template"]
        return None

    def switch_option_attaches(self, starting_list, route, product_smiles, option):
        """Dry run (switch choice): would `route` attach its step 1 with `option` forced? Leaves no
        lasting state: explored_reaction and the pending record are restored afterwards."""
        saved, pend = set(self.explored_reaction), getattr(self, "_switch_pending", None)
        self._switch_forced = {"product": product_smiles, "key": option["key"], "template": option["template"],
                               "reactants": option["reactants"]}
        try:
            checked, ev = self.sanitize(starting_list, copy.deepcopy(route), True)
            if not (ev and ev[0][1]):
                return False
            got = {sanitize_smiles(x) for x in ast.literal_eval(checked[0]["Reactants"])}
            return got == {sanitize_smiles(x) for x in option["reactants"]}
        except Exception:
            return False
        finally:
            self._switch_forced = None
            self.explored_reaction = saved
            self._switch_pending = pend

    def switch_options(self, reaction_smiles, product_smiles, model_reactants, exploration_signal, k=3):
        """Templates the silent switch could use for this step, in blurry_search's order (same
        candidate list; rule_based_search only if none of them applies), each with its top-ranked
        outcome and its closeness to the model's reactants (switch_choice.closeness). Returns
        {options: up to k NEAR-MISS outcomes (distinct), silent: what blurry_search would have
        taken, scanned, applicable, different}."""
        import switch_choice as sc
        out = {"options": [], "silent": None, "scanned": 0, "applicable": 0, "different": 0}
        seen = set()
        try:
            target_rd = rdchiralReactants(product_smiles)
        except Exception:
            return out

        def consider(smarts, key):
            out["scanned"] += 1
            if (product_smiles, smarts) in self.explored_reaction:
                return
            try:
                outs = run_retro(target_rd, smarts)
                if len(outs) > 1:
                    outs = self.rank_reactants(outs)
            except Exception:
                return
            if not outs or not outs[0]:
                return
            reac = [sanitize_smiles(r) or r for r in outs[0]]
            label, sim = sc.closeness(reac, model_reactants or [])
            rec = {"key": key, "template": smarts, "reactants": reac, "label": label, "sim": sim}
            out["applicable"] += 1
            if out["silent"] is None:
                out["silent"] = rec
            if label not in sc.CLOSE:
                out["different"] += 1
                return
            sig = frozenset(reac)
            if sig not in seen:
                seen.add(sig)
                out["options"].append(rec)

        try:
            rxn = smiles_to_reaction(self.sanitize_reaction(reaction_smiles))
            fp_re = rdChemReactions.CreateDifferenceFingerprintForReaction(rxn)
            sims = DataStructs.BulkTanimotoSimilarity(fp_re, self.all_reaction_fps)
            top = heapq.nlargest(1000 if exploration_signal else 100, range(len(sims)), key=lambda i: sims[i])
            candidates = [self.reaction_list[i] for i in top]
        except Exception:
            candidates = []
        for smarts in candidates:
            consider(smarts, self.template_to_key.get(smarts))
            if len(out["options"]) >= k:
                break
        if out["silent"] is None:                       # blurry_search would fall back to rule_based_search
            ok, key, smarts = self.rule_based_search(product_smiles, reaction_smiles)
            if ok:
                consider(smarts, key)
        return out



    
    def _switch_pending_outcome(self, step, data, reaction_outputs, template):
        """Switch choice, second silent path: step 1's own template applies but none of its outcomes is
        the model's reactant set (fix_reaction_error would write outcome 1 in). Pending options = this
        template's near-miss outcomes, then blurry_search's near misses."""
        import switch_choice as sc
        mine = list(data.get('reactants') or [])
        product = data['product'][0]
        try:
            own = (ast.literal_eval(step['Reaction']) or [None])[0]
        except Exception:
            own = None
        k = 2 * int(getattr(self, "_switch_max_options", 3))   # some fail the dry run
        opts, seen, silent = [], set(), None
        for out in reaction_outputs:
            reac = [sanitize_smiles(r) or r for r in out]
            label, sim = sc.closeness(reac, mine)
            rec = {"key": data.get('reaction_key'), "template": template, "reactants": reac, "label": label, "sim": sim}
            silent = silent or rec
            if label in sc.CLOSE and frozenset(reac) not in seen:
                seen.add(frozenset(reac))
                opts.append(rec)
        found = self.switch_options(own, product, mine, False, k=k) if own else {"options": []}
        for rec in found["options"]:
            if frozenset(rec["reactants"]) not in seen:
                seen.add(frozenset(rec["reactants"]))
                opts.append(rec)
        self._switch_pending = {"options": opts, "silent": silent, "scanned": found.get("scanned"),
                                "applicable": found.get("applicable"), "different": found.get("different"),
                                "product": product, "reason": "different_outcome", "reaction": own,
                                "reactants": mine}

    def sanitize(self, starting_list, route, exploration_signal):
        canon = bool(self.llm_config.get('canonical_product_match', False))
        new_route = check_and_update_routes(route, starting_list)
        if canon:
            self._fx_bump('spelling_fixed', self._align_product_spelling(new_route))
        first_evaluation = self.check_route(starting_list, new_route, exploration_signal)

        new_route = map_reaction(new_route, first_evaluation)
        new_route = self.fix_reaction_error(new_route, first_evaluation)
        new_route = check_and_update_routes(new_route, starting_list)
        if canon:
            # fix_reaction_error rewrote the molecule sets to canonical template
            # outputs, so re-align the model's product spellings before the final check.
            self._fx_bump('spelling_fixed_after_repair', self._align_product_spelling(new_route))

        final_evaluation = self.check_route_extra(starting_list, new_route, first_evaluation)

        return new_route, final_evaluation

    @staticmethod
    def _align_product_spelling(route):
        """Config `canonical_product_match` (default off = upstream behaviour).

        check_route / check_route_extra test `product[0] in molecule_set` as RAW
        strings, so a step whose product is the right molecule written as a
        different (non-canonical) SMILES is rejected. That was 11% of answers in the
        old baseline (8.5% in the 2026-09-27 27B smoke test). If the product is not
        literally in the step's molecule set but its canonical SMILES equals a
        member's, rewrite the product to that member's spelling. Only identical
        molecules match, so no wrong molecule can be admitted. Returns the number of
        rewrites (for the fix-effect counters).
        """
        n = 0
        for step in route:
            try:
                molecule_set = ast.literal_eval(step['Molecule set'])
                product = extract_molecules_from_output(step['Product'])
            except Exception:
                continue
            if not product or product[0] in molecule_set:
                continue
            target = sanitize_smiles(product[0])
            if target is None:
                continue
            for member in molecule_set:
                if sanitize_smiles(member) == target:
                    step['Product'] = str([member] + list(product[1:]))
                    n += 1
                    break
        return n

    def sort_buffer(self):
        self.oracle.sort_buffer()
    
    def log_intermediate(self, finish=False):
        self.oracle.log_intermediate(finish=finish)
    

        
    def save_result(self, suffix=None):

        print(f"Saving...")

        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            suffix = suffix.replace("/", "")
            output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + '.yaml')

        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.route_buffer, f, sort_keys=False)

    def write_done_marker(self, suffix, n_attempts=1, solved_on_attempt=None,
                          termination=None):
        """Mark a molecule as finished. Writes two sidecars next to the yaml:
          results_{suffix}.done         - zero-byte flag (cheap glob)
          results_{suffix}.status.json  - structured metadata
        Termination reason inferred from final oracle state. `n_attempts` /
        `solved_on_attempt` record the retry-to-solve loop (Run S v2).
        """
        suffix = suffix.replace("/", "")
        base = os.path.join(self.args.output_dir, 'results_' + suffix)
        buf = self.route_buffer
        if not buf:
            # RESUME-FINALIZE FIX (code review 2026-11-22). optimize()'s "already finished in a prior run
            # but .done missing" branch calls us AFTER reset() has emptied route_buffer, so an in-memory
            # read reported passed=False / n_routes=0 / EARLY_TERM for a molecule that had actually PASSED.
            # save_result is skipped on that path, so the real yaml survives and contradicts its own
            # sidecar -- and .status.json is what every analysis script counts, so this silently
            # UNDERCOUNTED passes on any walltime-cut retry arm. Recover the scores from the yaml.
            try:
                with open(base + '.yaml') as f:
                    buf = yaml.safe_load(f) or {}
            except FileNotFoundError:
                buf = {}
            except Exception as e:
                print(f"[done-marker] could not read back {base}.yaml: {type(e).__name__} {e}")
                buf = {}
        scores = [float(v[0]) for v in buf.values()] if buf else []
        passed = any(s == 0.0 for s in scores)
        n_routes = len(buf)
        max_oracle = self.oracle.max_oracle_calls
        if passed:
            status = "PASS"
            termination = "found_pass"
        elif n_routes >= max_oracle:
            status = "BUDGET"
            termination = "max_oracle_calls"
        else:
            status = "EARLY_TERM"
            # Was hardcoded "max_evolution_iters" (never measured); use the
            # cause optimize() observed ("time_cap" / "budget_or_iters").
            termination = termination or "unknown"
        pass_steps = None
        if passed:
            ps = [int(v[1]) for v in buf.values() if float(v[0]) == 0.0]
            if ps:
                pass_steps = min(ps)
        meta = {
            "status": status,
            "termination": termination,
            "passed": passed,
            "n_routes": n_routes,
            "best_score": max(scores) if scores else None,
            "pass_steps": pass_steps,
            "max_oracle_calls": max_oracle,
            "n_attempts": n_attempts,
            "solved_on_attempt": solved_on_attempt,
            "completed_at": _datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "pid": os.getpid(),
        }
        with open(base + ".status.json", "w") as f:
            json.dump(meta, f, indent=2)
        # zero-byte flag, written last so its existence implies status.json is also there
        open(base + ".done", "w").close()

    def _lookup_template(self, reaction, product, reactants):
        """Template lookup for check_route. Default = upstream is_reaction_in_dict:
        the FIRST template whose SMARTS match the molecules.

        Config `prefer_reproducing_template` (2026-10-04): a step typically matches
        ~14 templates (max 59) with different outcomes, and upstream keeps the first
        in dict order. Of 75 known-correct routes that broke at a later step, 56
        broke because a different matching template reproduces the route's actual
        reactants. With the flag on, use the first matching template whose outcome
        IS the proposed reactant set; if none does, fall back to the first match."""
        if not self.llm_config.get('prefer_reproducing_template', False):
            return is_reaction_in_dict(reaction, self.template_dict)
        keys = _all_matching_template_keys(reaction, self.template_dict)
        if not keys:
            return False, None
        try:
            model = {sanitize_smiles(r) for r in reactants}
            target_rd = rdchiralReactants(product[0])
        except Exception:
            return True, keys[0]
        self._fx_bump('tmpl_lookups')
        for k in keys:
            try:
                for out in run_retro(target_rd, self.original_template_dict[k]):
                    if {sanitize_smiles(x) for x in out} == model:
                        if k != keys[0]:
                            self._fx_bump('tmpl_fix_changed')   # upstream would have used keys[0]
                        return True, k
            except Exception:
                continue
        self._fx_bump('tmpl_no_reproducer')
        return True, keys[0]

    def check_route(self, target_smi, route, exploration_signal):
        """
        Check if the route is valid
        target smi is a list
        """
        results = []
        for i in range(len(route)):
            current_step_index = i
            current_step = route[current_step_index]

            step_validity = False
            molecule_set = ast.literal_eval(current_step['Molecule set'])
            updated_molecule_set = ast.literal_eval(current_step['Updated molecule set'])
            reaction = ast.literal_eval(current_step['Reaction'])[0]
            #reaction = change_to_forward_reaction(reaction)
            product = extract_molecules_from_output(current_step['Product'])
            reactants = ast.literal_eval(current_step['Reactants'])

            #step 1: check molecules' validity
            starting_signal = True
            if current_step_index == 0:
                mdd = set(molecule_set).issubset(set(target_smi))
                if not mdd:
                    starting_signal = False

            # Product in molecule set
            product_inside = False
            if product[0] in molecule_set:
                product_inside = True
            invalid_molset_mol_id = []
            invalid_updated_mol_id = []
            
            updated_set_signals = check_validity(updated_molecule_set)
            if False in updated_set_signals:
                invalid_updated_mol_id = [index for index, value in enumerate(updated_set_signals) if not value]
    
            mol_set_signals = check_validity(molecule_set)
            if False in mol_set_signals:
                invalid_molset_mol_id = [index for index, value in enumerate(mol_set_signals) if not value]

            #check purchasability
            check_availability = False
            unavailable_mol_id = []
            if i == len(route) - 1:
                availibities = check_purchasable(updated_molecule_set, updated_set_signals, self.inventory)
                check_availability = True
            if check_availability == True:
                if False in availibities:
                    unavailable_mol_id = [index for index, value in enumerate(availibities) if not value]
    
            #step 2
            reaction_valid, updated_set_valid, reaction_existance = False, False, False
            if ':' in reaction:
                keys = [key for key, value in self.original_template_dict.items() if value == reaction]
                if len(keys) == 1:
                    reaction_existance = True
                    reaction_key = keys[0]
                else:
                    reaction_existance = False
                    reaction_key = None
            else:
                if current_step_index == 0:
                    reaction_existance, reaction_key = self._lookup_template(reaction, product, reactants)
                    #reaction_existance, reaction_key, new_reaction = self.blurry_search(reaction, product[0], exploration_signal)
                #To save time, we only check the step after a valid step
                elif results[-1][1] == True:
                    reaction_existance, reaction_key = self._lookup_template(reaction, product, reactants)

                else:
                    reaction_existance = False
                    reaction_key = None
            if reaction_key == None:
                new_reaction = reaction
            else:
                new_reaction = self.original_template_dict[reaction_key]
                
            if reaction_existance == True:
                reaction_valid, updated_set_valid = verify_reaction_step(molecule_set, updated_molecule_set, new_reaction, product, reactants, self.inventory, self.oracle)
            
            if current_step_index == 0:
                # AOT switch choice: the model's pick replaces the lookup (None unless switch choice is on);
                # _step1_substitute = upstream blurry_search unless AOT switch choice is on
                forced = self._switch_forced_for(product[0])
                if forced:
                    reaction_existance, reaction_key, new_reaction = forced
                elif reaction_key == None:
                    reaction_existance, reaction_key, new_reaction = self._step1_substitute(
                        reaction, product[0], reactants, exploration_signal, "no_template")
                elif (product[0], new_reaction) in self.explored_reaction:
                    reaction_existance, reaction_key, new_reaction = self._step1_substitute(
                        reaction, product[0], reactants, exploration_signal, "already_used")
                elif reaction_existance == True and reaction_valid == False:
                    reaction_existance, reaction_key, new_reaction = self._step1_substitute(
                        reaction, product[0], reactants, exploration_signal, "not_applicable")

                if reaction_key == None:
                    new_reaction = reaction
                else:
                    #new_reaction = self.original_template_dict[reaction_key]
                    reaction_valid, updated_set_valid = verify_reaction_step(molecule_set, updated_molecule_set, new_reaction, product, reactants, self.inventory, self.oracle)
               
            if (
                len(invalid_molset_mol_id) == 0 and
                len(invalid_updated_mol_id) == 0 and
                reaction_valid and
                updated_set_valid and
                starting_signal and
                product_inside
            ):
                step_validity = True

            # Construct the dictionary
            step_info = {
                "target_smi": target_smi,
                "starting_signal": starting_signal,
                "product_inside": product_inside,
                "molecule_set": molecule_set,
                "updated_molecule_set": updated_molecule_set,
                "reaction": new_reaction,
                "reaction_key": reaction_key,
                "product": product,
                "reactants": reactants,
                "updated_set_signals": updated_set_signals,
                "invalid_updated_mol_id": invalid_updated_mol_id,
                "mol_set_signals": mol_set_signals,
                "invalid_molset_mol_id": invalid_molset_mol_id,
                "check_availability": check_availability,
                "unavailable_mol_id": unavailable_mol_id,
                "reaction_existance": reaction_existance,
                "reaction_valid": reaction_valid,
                "updated_set_valid": updated_set_valid
            }

            # Store the tuple in the results list
            results.append((current_step_index, step_validity, step_info))
        return results

    def check_route_extra(self, target_smi, route, first_evaluation):
        """
        Check if the route is valid
        """
        results = []
        for i in range(len(route)):
            current_step_index = i
            current_step = route[current_step_index]
            step_id, is_valid, current_evaluation = first_evaluation[current_step_index]

            step_validity = False
            molecule_set = ast.literal_eval(current_step['Molecule set'])
            updated_molecule_set = ast.literal_eval(current_step['Updated molecule set'])
            reaction = ast.literal_eval(current_step['Reaction'])[0]
            product = extract_molecules_from_output(current_step['Product'])
            reactants = ast.literal_eval(current_step['Reactants'])

            #step 1: check molecules' validity
            starting_signal = True
            if current_step_index == 0:
                mmd = set(molecule_set).issubset(set(target_smi))
                if not mmd:
                    starting_signal = False

            # Product in molecule set
            product_inside = False
            if product[0] in molecule_set:
                product_inside = True

            invalid_molset_mol_id = []
            invalid_updated_mol_id = []
            
            updated_set_signals = check_validity(updated_molecule_set)
            if False in updated_set_signals:
                invalid_updated_mol_id = [index for index, value in enumerate(updated_set_signals) if not value]
    
            mol_set_signals = check_validity(molecule_set)
            if False in mol_set_signals:
                invalid_molset_mol_id = [index for index, value in enumerate(mol_set_signals) if not value]


            check_availability = False
            unavailable_mol_id = []
            if i == len(route) - 1:
                availibities = check_purchasable(updated_molecule_set, updated_set_signals, self.inventory)
                check_availability = True
            if check_availability == True:
                if False in availibities:
                    unavailable_mol_id = [index for index, value in enumerate(availibities) if not value]
    
            #step 2
            reaction_valid, updated_set_valid = False, False
            reaction_existance, reaction_key = current_evaluation['reaction_existance'], current_evaluation['reaction_key']
            
            new_reaction = current_evaluation["reaction"]

            if reaction_existance == True:
                reaction_valid, updated_set_valid = verify_reaction_step(molecule_set, updated_molecule_set, new_reaction, product, reactants, self.inventory, self.oracle)
    
    
            if (
                len(invalid_molset_mol_id) == 0 and
                len(invalid_updated_mol_id) == 0 and
                reaction_valid and
                updated_set_valid and
                starting_signal and
                product_inside
            ):
                step_validity = True
            if step_validity == True:
                self.explored_reaction.add((product[0], new_reaction))
            # Construct the dictionary
            step_info = {
                "target_smi": target_smi,
                "starting_signal": starting_signal,
                "product_inside": product_inside,
                "molecule_set": molecule_set,
                "updated_molecule_set": updated_molecule_set,
                "reaction": new_reaction,
                "reaction_key": reaction_key,
                "product": product,
                "reactants": reactants,
                "updated_set_signals": updated_set_signals,
                "invalid_updated_mol_id": invalid_updated_mol_id,
                "mol_set_signals": mol_set_signals,
                "invalid_molset_mol_id": invalid_molset_mol_id,
                "check_availability": check_availability,
                "unavailable_mol_id": unavailable_mol_id,
                "reaction_existance": reaction_existance,
                "reaction_valid": reaction_valid,
                "updated_set_valid": updated_set_valid
            }

            # Store the tuple in the results list
            results.append((current_step_index, step_validity, step_info))
        return results

    def reset(self):
        del self.oracle
        self.oracle = Oracle(args=self.args)
        self.oracle.route_buffer = {}
        self.oracle.reaction_cache = dict()
        self.explored_reaction = set()
        self.visited_molecules = dict()
        # Upstream main.py built a NEW optimizer per molecule, so dead_molecules
        # never outlived one target. We reuse one optimizer per worker, so without
        # this a molecule marked dead on target i kept a +100 penalty on target j.
        self.dead_molecules = dict()
        self._fx = {}
        # arm `ask` (2026-11-15): the stall counter, the advisor's notes, its call count and any
        # pending destination are all PER TARGET. We reuse one optimizer per worker, so without
        # this a note written on target i would be shown to the advisor on target j.
        self._ask_dry = 0
        self._ask_calls = 0
        self._ask_notes = []
        self._ask_recent = []
        self._ask_pending = None

    def _fx_bump(self, key, n=1):
        """Fix-effect counters (2026-10-07): how often each config-gated fix actually
        changed something for the current molecule. Logging only; never alters results.
        Written to the AOT status.json as `fix_stats`."""
        d = self.__dict__.setdefault('_fx', {})
        d[key] = d.get(key, 0) + (n or 0)

    # ------------------------------------------------------------------
    # Walltime-resume checkpoint (2026-09-26). A molecule cut by the 12h PBS
    # walltime used to restart from scratch next chunk and only the LAST attempt
    # was recorded, so walltime-cut molecules got several fresh 50-call tries
    # (25/54 baseline passes were on a restarted molecule). The GA state is now
    # checkpointed after every init slot and every GA iteration and restored on
    # restart, so the oracle budget and the time cap are cumulative per attempt.
    # Work done after the last checkpoint is redone (the buffer rolls back with
    # it, so the budget stays exact; that wall-clock is not charged to the cap).
    # ------------------------------------------------------------------
    def _ckpt_save(self, stage, **state):
        path = getattr(self, '_ckpt_path', None)
        if not path:
            return
        state.update(
            stage=stage,
            attempt=getattr(self, '_ckpt_attempt', 1),
            elapsed_s=_time.time() - self._attempt_start,
            route_buffer=self.oracle.route_buffer,
            reaction_cache=self.oracle.reaction_cache,
            last_log=self.oracle.last_log,
            max_oracle_calls=self.oracle.max_oracle_calls,
            explored_reaction=self.explored_reaction,
            visited_molecules=self.visited_molecules,
            dead_molecules=self.dead_molecules,
        )
        tmp = path + '.tmp'
        try:
            with open(tmp, 'wb') as f:
                pickle.dump(state, f)
            os.replace(tmp, path)  # atomic: a kill mid-write leaves the old ckpt
        except Exception as e:
            print(f"[ckpt] save failed (non-fatal): {type(e).__name__} {e}")

    def _ckpt_load(self, attempt):
        path = getattr(self, '_ckpt_path', None)
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
        except Exception as e:
            print(f"[ckpt] unreadable, starting fresh: {type(e).__name__} {e}")
            return None
        if state.get('attempt') != attempt:
            return None
        return state

    def _ckpt_restore_common(self, state):
        self.oracle.route_buffer = state['route_buffer']
        self.oracle.reaction_cache = state['reaction_cache']
        self.oracle.last_log = state['last_log']
        self.oracle.max_oracle_calls = state['max_oracle_calls']
        self.explored_reaction = state['explored_reaction']
        self.visited_molecules = state['visited_molecules']
        self.dead_molecules = state['dead_molecules']

    def _ckpt_clear(self):
        path = getattr(self, '_ckpt_path', None)
        if path and os.path.exists(path):
            os.remove(path)

    @property
    def route_buffer(self):
        return self.oracle.route_buffer

    @property
    def finish(self):
        return self.oracle.finish
        
    def _optimize(self, oracle, config):
        raise NotImplementedError
            
    def rewards(self, route_evaluation):
        return self.oracle(self.inventory, route_evaluation, self.visited_molecules, self.dead_molecules)

    def build_full_evaluation(self, route_item, extension_evaluation):
        """Rebuild a modification's evaluation as the FULL route from the true
        target, not the extension fragment alone.

        modification() validates the LLM's new route against the parent's
        leftover leaf set, so `extension_evaluation` starts at intermediate
        molecules. Keying the oracle buffer on that fragment recorded passes
        for "routes" that never contain the target; the issubset check in
        check_route also let the fragment silently drop unpurchasable leaves.
        The parent's already-validated chain is prepended as valid steps and
        the final purchasability check is recomputed over the aligned leaf set
        (which carries unconsumed leftovers), so a 0.0 score now means the
        whole tree under the target is purchasable.
        """
        target = route_item.target_smiles
        target = [target] if isinstance(target, str) else list(target)
        chain = route_item.validated_route
        n_appended = 0
        for _, ok, _ in extension_evaluation:
            if not ok:
                break
            n_appended += 1
        prefix = chain[:len(chain) - n_appended] if n_appended else chain

        def _info(step, last):
            molecule_set = list(step['Molecule set'])
            updated = list(step['Updated molecule set'])
            reaction = step['Reaction']
            if isinstance(reaction, list):
                reaction = reaction[0] if reaction else ""
            info = {
                "target_smi": target,
                "starting_signal": True,
                "product_inside": True,
                "molecule_set": molecule_set,
                "updated_molecule_set": updated,
                "reaction": reaction,
                "reaction_key": None,
                "product": list(step.get('Product') or []),
                "reactants": list(step.get('Reactants') or []),
                "updated_set_signals": [True] * len(updated),
                "invalid_updated_mol_id": [],
                "mol_set_signals": [True] * len(molecule_set),
                "invalid_molset_mol_id": [],
                "check_availability": False,
                "unavailable_mol_id": [],
                "reaction_existance": True,
                "reaction_valid": True,
                "updated_set_valid": True,
            }
            if last:
                signals = check_validity(updated)
                avail = check_purchasable(updated, signals, self.inventory)
                info["check_availability"] = True
                info["unavailable_mol_id"] = [i for i, v in enumerate(avail) if not v]
            return info

        full = []
        for step in prefix:
            if 'Molecule set' not in step:  # init-fallback stub: no validated steps yet
                continue
            full.append((len(full), True, _info(step, last=False)))
        ext_all_valid = n_appended == len(extension_evaluation)
        aligned = chain[len(chain) - n_appended:] if n_appended else []
        for j, step in enumerate(aligned):
            last = ext_all_valid and j == n_appended - 1
            full.append((len(full), True, _info(step, last=last)))
        for k in range(n_appended, len(extension_evaluation)):
            _, ok, info = extension_evaluation[k]
            full.append((len(full), ok, info))
        if not full:
            return extension_evaluation
        return full
    
    def update_cache(self, mol_smiles, reaction):
        try:
            s = rdChemReactions.CreateDifferenceFingerprintForReaction(smiles_to_reaction(reaction))
            self.oracle.store_cache(mol_smiles, reaction)
        except:
            pass


    def _attempts_path(self, suffix):
        return os.path.join(self.args.output_dir,
                            'results_' + suffix.replace('/', '') + '.attempts.json')

    def _read_attempts(self, suffix):
        """Per-molecule retry progress persisted across job runs (Run S v2)."""
        try:
            with open(self._attempts_path(suffix)) as f:
                return json.load(f)
        except Exception:
            return {"attempts_done": 0, "solved": False, "solved_on_attempt": None}

    def _write_attempts(self, suffix, d):
        p = self._attempts_path(suffix)
        tmp = p + f".tmp.{os.getpid()}"
        try:
            with open(tmp, "w") as f:
                json.dump(d, f, indent=2)
            os.replace(tmp, p)
        except Exception as e:
            print(f"[attempts] write failed: {type(e).__name__} {e}")

    def _append_attempt_log(self, target, attempt, max_attempts, passed, term):
        """Emit a per-attempt fail/pass log line (each attempt, every run)."""
        idx = getattr(self, 'tb_target_idx', -1)
        status = "PASS" if passed else "FAIL"
        print(f"[attempt] idx={idx} attempt {attempt}/{max_attempts} -> {status} ({term})")
        try:
            p = os.path.join(self.args.output_dir, 'attempt_log.jsonl')
            with _APPEND_LOCK, open(p, "a") as f:
                f.write(json.dumps({
                    "ts": _datetime.utcnow().isoformat() + "Z", "pid": os.getpid(),
                    "target_idx": idx, "target_smi": target,
                    "attempt": attempt, "max_attempts": max_attempts,
                    "status": status, "termination": term,
                    "n_routes": len(self.oracle.route_buffer),
                }) + "\n")
        except Exception:
            pass

    def optimize(self, target, route_list, all_fps, config, seed=0, project="test",
                 tb_writer=None, target_idx=0, ground_truth=None):
        self.reset()
        self.seed = seed
        self.tb_writer = tb_writer
        self.tb_target_idx = target_idx
        self.oracle.task_label = self.model_name + "_" + target + "_" + str(seed)
        self._attempt_deadline = None   # inert unless a per-attempt cap is set

        # Per-molecule reload of strategy guide. Without this, a long-lived worker
        # would keep using the guide it loaded at __init__ and never see updates
        # made by other workers (or by itself on prior molecules).
        guide_path = self.llm_config.get('strategy_guide_path', '')
        if guide_path and os.path.exists(guide_path):
            try:
                with open(guide_path) as _gf:
                    self._strategy_context = _gf.read()
            except Exception:
                pass

        suffix = self.model_name + "_" + target + "_" + str(seed)

        # Retry-to-solve loop (Run S v2): attempt the molecule up to max_attempts
        # times, stopping early on PASS. Skills are extracted after EVERY attempt
        # (success and failure). Each attempt is wall-clock capped
        # (attempt_time_limit_hours) so a brutal failing attempt can't eat a whole
        # chunk. Progress persists across job runs (.attempts.json): a molecule that
        # did 2/4 attempts before walltime resumes at attempt 3 next run instead of
        # restarting. For non-skill runs max_attempts=1 -> behaviour unchanged.
        max_attempts = 1
        # TOP-LEVEL cap, applies to every run mode (A/B/C/D/S). It used to live
        # only inside the skill_memory branch below, so the clean baseline (which
        # deliberately has NO skill_memory block) got no cap at all: a worker that
        # hit a pathological molecule ground to walltime, was killed, resumed on
        # the SAME molecule next chunk, and never advanced. Measured 2026-08-01 on
        # pistachio_hard_clean_postfix -- workers 1 and 5 were still on their FIRST
        # molecule (idx 10 / idx 50) after 10 days, so 20 of 100 molecules had
        # never been attempted. skill_memory.attempt_time_limit_hours still takes
        # precedence when present, so every existing Run S arm is unchanged.
        time_limit_h = float(self.llm_config.get('attempt_time_limit_hours', 0) or 0)
        if self._skill_memory_enabled:
            max_attempts = max(1, int(self._skill_cfg.get('max_attempts', 1)))
            time_limit_h = float(
                self._skill_cfg.get('attempt_time_limit_hours', time_limit_h) or time_limit_h)

        start_attempt = 1
        solved_on_attempt = None
        if max_attempts > 1:
            prog = self._read_attempts(suffix)
            if prog.get("solved") or prog.get("attempts_done", 0) >= max_attempts:
                # Already finished in a prior run but .done missing (e.g. killed
                # right after the last attempt) — finalize and skip re-running.
                try:
                    self.write_done_marker(
                        suffix, n_attempts=prog.get("attempts_done", max_attempts),
                        solved_on_attempt=prog.get("solved_on_attempt"))
                except Exception as e:
                    print(f"[done-marker] non-fatal failure: {type(e).__name__} {e}")
                return
            start_attempt = prog.get("attempts_done", 0) + 1
            solved_on_attempt = prog.get("solved_on_attempt")
            if start_attempt > 1:
                print(f"[resume] idx={target_idx} resuming at attempt "
                      f"{start_attempt}/{max_attempts} (prior run did "
                      f"{start_attempt - 1})")

        n_attempts = start_attempt - 1
        term = None
        for attempt in range(start_attempt, max_attempts + 1):
            n_attempts = attempt
            if attempt > 1:
                # Fresh GA attempt: clear the oracle/route buffer.
                self.reset()
                self.oracle.task_label = self.model_name + "_" + target + "_" + str(seed)
            # Per-attempt reload of skill memory (sees other workers' / prior
            # attempts' updates) + reset the cached bandit selection so each
            # attempt gets a fresh Thompson draw.
            self._injected_skill_ids = []
            self._injected_skill_block = None
            self._injected_skill_ids_all = set()   # union across init + all mods
            self._relevance_cache = None            # per-molecule relevance map
            self._rag_cov_cache = None               # per-molecule RAG-coverage memo
            self._retron_cache = None                 # per-molecule retron-match memo
            self._precq_cache = None                  # per-molecule rdchiral/SA/purch memo
            self._purch_cache = None                  # per-molecule purchasability memo
            self._pool_cache = None
            self._current_target = target           # for stall-gated mid-run selection
            if self._skill_memory_enabled:
                self._skill_memory = self._sm.load_memory(self._skill_memory_path)

            # Resume a walltime-cut attempt from its checkpoint, if any.
            self._ckpt_path = os.path.join(
                self.args.output_dir, 'results_' + suffix.replace("/", "") + '.ckpt.pkl')
            self._ckpt_attempt = attempt
            resume = self._ckpt_load(attempt)
            self._resume_state = resume
            if resume is not None:
                progress = (f"{resume['llm_calls']} LLM calls" if 'llm_calls' in resume
                            else f"{len(resume['route_buffer'])} routes")
                print(f"[ckpt] idx={target_idx} resuming attempt {attempt} at stage="
                      f"{resume['stage']} with {progress}, "
                      f"{resume['elapsed_s']/3600:.2f}h already used")

            # Per-attempt wall-clock cap (read by the GA loop in run.py). Elapsed
            # time from a resumed checkpoint counts toward it.
            self._attempt_start = _time.time() - (resume['elapsed_s'] if resume else 0.0)
            self._attempt_deadline = (self._attempt_start + time_limit_h * 3600
                                      if time_limit_h > 0 else None)

            self._optimize(target, route_list, all_fps, config)
            # Attempt finished normally: its checkpoint must not seed the next
            # attempt or a re-run. (An exception leaves it in place to resume.)
            self._ckpt_clear()

            passed = any(v[0] == 0 for v in self.oracle.route_buffer.values())
            elapsed_h = (_time.time() - self._attempt_start) / 3600
            term = ("pass" if passed else
                    ("time_cap" if (time_limit_h > 0 and elapsed_h >= time_limit_h * 0.99)
                     else "budget_or_iters"))
            is_final = passed or (attempt == max_attempts)
            # freeze: use a pre-built store read-only (skip run-time extraction/
            # credit/lifecycle). For testing a curated store (e.g. USPTO-derived)
            # without diluting it with run-time skills.
            if self._skill_memory_enabled and not self._skill_cfg.get('freeze', False):
                try:
                    self._update_skill_memory(target, attempt=attempt,
                                              ground_truth=ground_truth,
                                              is_final=is_final)
                except Exception as e:
                    print(f"[skill-memory] non-fatal failure: {type(e).__name__} {e}")

            if passed:
                solved_on_attempt = attempt

            # Persist progress + emit the per-attempt fail/pass log (after the
            # attempt completes, so a walltime kill mid-attempt re-does only that
            # attempt next run, never a completed one).
            if max_attempts > 1:
                self._write_attempts(suffix, {
                    "attempts_done": attempt, "solved": bool(passed),
                    "solved_on_attempt": solved_on_attempt,
                    "max_attempts": max_attempts,
                    "last_status": "PASS" if passed else "FAIL",
                    "last_termination": term,
                    "updated_at": _datetime.utcnow().isoformat() + "Z",
                })
                self._append_attempt_log(target, attempt, max_attempts, passed, term)

            if passed:
                break

        self._attempt_deadline = None

        # Log per-target result to TensorBoard
        if tb_writer is not None:
            solved = int(any(v[0] == 0 for v in self.oracle.route_buffer.values()))
            oracle_calls_used = len(self.oracle.route_buffer)
            tb_writer.add_scalar('target/solved', solved, target_idx)
            tb_writer.add_scalar('target/oracle_calls_used', oracle_calls_used, target_idx)
            tb_writer.flush()

        if self.args.log_results:
            self.log_intermediate(finish=True)
        self.save_result(suffix)
        # Mark the molecule as truly finished ONLY when solved or all attempts are
        # exhausted. A partial molecule (e.g. 2/4 attempts) is intentionally left
        # WITHOUT a .done so the next run resumes it at the next attempt.
        if solved_on_attempt is not None or n_attempts >= max_attempts:
            try:
                self.write_done_marker(suffix, n_attempts=n_attempts,
                                       solved_on_attempt=solved_on_attempt,
                                       termination=term)
            except Exception as e:
                print(f"[done-marker] non-fatal failure: {type(e).__name__} {e}")

        if self.llm_config.get('strategy_guide_update', False):
            try:
                self._update_strategy_guide(target)
            except Exception as e:
                print(f"[guide-update] non-fatal failure: {type(e).__name__} {e}")

    # ------------------------------------------------------------------
    # Self-updating strategy guide (Run A guided_v3).
    # See the serialized-scoring design note.
    # ------------------------------------------------------------------

    def _step_status_tag(self, step):
        """Deterministic per-step status classifier — see plan Section 5."""
        try:
            _step_id, valid, data = step
        except Exception:
            return "malformed"
        if not data.get("reaction_existance", True):
            return "reaction_not_in_USPTO"
        if not data.get("reaction_valid", True):
            return "invalid_template_application"
        unavail = data.get("unavailable_mol_id") or []
        if unavail:
            return f"unpurchasable_leaves:{list(unavail)}"
        if data.get("check_availability") and not unavail:
            return "ok_purchasable_leaf"
        return "partial_progress"

    def _describe_target(self, target):
        """Structural profile for the skill-relevance ranker. Includes the SMILES
        and the target's functional groups -- the ranker's whole job is to judge
        which disconnection skills apply, and it cannot do that blind to the
        reactive groups present. (Earlier versions withheld the SMILES; that
        starved the ranker of exactly the FG signal it needs.)"""
        try:
            mol = Chem.MolFromSmiles(target)
            if mol is None:
                return f"TARGET SMILES: {target}\nTARGET PROFILE: (unparseable)"
            heavy = mol.GetNumHeavyAtoms()
            rings = mol.GetRingInfo().NumRings()
            aromatic_rings = sum(
                1 for ring in mol.GetRingInfo().AtomRings()
                if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
            )
            stereo = sum(1 for a in mol.GetAtoms() if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED)
            rot = AllChem.CalcNumRotatableBonds(mol)
            halogens = sum(1 for a in mol.GetAtoms() if a.GetSymbol() in ("F", "Cl", "Br", "I"))
            heteros = sum(1 for a in mol.GetAtoms() if a.GetSymbol() not in ("C", "H"))
            fgs = ""
            try:
                from skill_matcher import fg_set
                names = sorted(fg.replace("fr_", "") for fg in fg_set(target))
                if names:
                    fgs = "\nFUNCTIONAL GROUPS: " + ", ".join(names)
            except Exception:
                pass
            return (
                f"TARGET SMILES: {target}\n"
                f"TARGET PROFILE: heavy_atoms={heavy}, rings={rings}, "
                f"aromatic_rings={aromatic_rings}, stereocenters={stereo}, "
                f"rotatable_bonds={rot}, halogens={halogens}, n_heteroatoms={heteros}"
                f"{fgs}"
            )
        except Exception as e:
            return f"TARGET SMILES: {target}\nTARGET PROFILE: (analysis failed: {type(e).__name__})"

    # ------------------------------------------------------------------
    # Ground-truth weighting (Run S v2). The dataset precursor is a single-step
    # disconnection of the target; we compare it to the FIRST disconnection of a
    # proposed route (the step that produces the target) by Morgan-fingerprint
    # Tanimoto over the reactant sets. 1.0 = exact (canonical) match.
    # ------------------------------------------------------------------

    @staticmethod
    def _split_smiles(smi):
        """Split a possibly dotted SMILES into canonical component SMILES."""
        out = []
        for part in (smi or "").replace(">>", ".").split("."):
            part = part.strip()
            if not part:
                continue
            m = Chem.MolFromSmiles(part)
            if m is not None:
                out.append(Chem.MolToSmiles(m))
        return out

    def _all_route_reactants(self, route_steps):
        """Every reactant that appears ANYWHERE in the route (across all steps),
        de-duplicated as canonical SMILES. This is the 'reach-the-precursors'
        pool: it catches the ground-truth precursor whether the route uses it as a
        final purchasable leaf or as an intermediate at any depth."""
        seen = []
        seen_set = set()
        for step in route_steps:
            try:
                _sid, _valid, data = step
            except Exception:
                continue
            for r in (data.get("reactants", []) or []):
                for canon in self._split_smiles(r):
                    if canon not in seen_set:
                        seen_set.add(canon)
                        seen.append(canon)
        return seen

    def _gt_similarity(self, route_steps, target, gt_precursor):
        """'Reach-the-precursors' similarity in [0,1]: how well the ground-truth
        precursor set is matched by ANY reactant appearing anywhere in the route
        (not just the first disconnection). Rewards a route that arrives at the
        known building blocks even over several steps. 1.0 if every ground-truth
        fragment appears (canonically) somewhere in the route; otherwise the
        average best-Tanimoto of each ground-truth fragment vs the route's whole
        reactant pool. 0.0 if either side is missing/unparseable. `target` is
        unused (kept for signature stability)."""
        if not gt_precursor or not route_steps:
            return 0.0
        try:
            route_reactants = self._all_route_reactants(route_steps)
            gt_reactants = self._split_smiles(gt_precursor)
            if not route_reactants or not gt_reactants:
                return 0.0
            # Exact: all known building blocks are reached somewhere in the route.
            if set(gt_reactants).issubset(set(route_reactants)):
                return 1.0
            route_fps = [AllChem.GetMorganFingerprint(Chem.MolFromSmiles(s), 2)
                         for s in route_reactants if Chem.MolFromSmiles(s)]
            gt_fps = [AllChem.GetMorganFingerprint(Chem.MolFromSmiles(s), 2)
                      for s in gt_reactants if Chem.MolFromSmiles(s)]
            if not route_fps or not gt_fps:
                return 0.0
            # Best pairwise match averaged over ground-truth components.
            sims = []
            for g in gt_fps:
                sims.append(max(DataStructs.TanimotoSimilarity(g, r) for r in route_fps))
            return float(max(0.0, min(1.0, sum(sims) / len(sims))))
        except Exception:
            return 0.0

    def _build_example(self, route_steps, target, outcome, gt_sim, idx):
        """Build a concrete worked example dict from a route's steps (exact SMILES,
        no LLM hallucination). Returns None if there are no usable steps."""
        if not route_steps:
            return None
        step_lines = []
        for step in route_steps:
            try:
                step_id, _valid, data = step
            except Exception:
                continue
            reaction = (data.get("reaction") or "").strip()
            if len(reaction) > 120:
                reaction = reaction[:117] + "..."
            tag = self._step_status_tag(step)
            step_lines.append(f"step {step_id}: {reaction}  [{tag}]")
        if not step_lines:
            return None
        return {
            "target_smiles": target,
            "steps": step_lines,
            "reasoning": "",
            "from_idx": idx,
            "gt_similarity": round(float(gt_sim), 3),
            "outcome": outcome,
        }

    def _gt_rank_routes(self, ground_truth, top_n=3):
        """Rank EVERY route the GA explored this attempt (the whole modification
        trajectory, held in oracle.route_buffer) by ground-truth closeness
        (reach-the-precursors). Returns the top_n as (steps, gt_sim, outcome,
        oracle_score), best-gt first. This is how skills learn from ALL GA
        modification attempts — successes and failures — weighted by similarity to
        the real synthesis (no floor; pure failures rank low but still appear)."""
        rb = self.oracle.route_buffer or {}
        scored = []
        for key, val in rb.items():
            try:
                steps = json.loads(key)
                osc = float(val[0])
            except Exception:
                continue
            gt = self._gt_similarity(steps, None, ground_truth)
            scored.append((steps, gt, "PASS" if osc == 0.0 else "FAIL", osc))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:max(1, top_n)]

    def _render_route(self, route_eval, header):
        """Render one route as numbered step lines with status tags."""
        lines = [header]
        for step in route_eval:
            try:
                step_id, _valid, data = step
            except Exception:
                lines.append(f"  (malformed step)")
                continue
            n_react = len(data.get("reactants", []) or [])
            n_prod = len(data.get("product", []) or [])
            reaction = (data.get("reaction") or "").strip()
            if len(reaction) > 90:
                reaction = reaction[:87] + "..."
            tag = self._step_status_tag(step)
            lines.append(
                f"  step {step_id}: {n_react} reactant(s) -> {n_prod} product(s)  "
                f"reaction={reaction}  [{tag}]"
            )
        return "\n".join(lines)

    def _build_mol_summary(self, target):
        """Compact, generalisable per-molecule summary fed to the update LLM.

        Reads self.oracle.route_buffer (already populated and sorted by score
        desc via save_result -> sort_buffer). Returns a text block + a small
        stats dict for logging.
        """
        rb = self.oracle.route_buffer or {}
        parsed = []
        for key, val in rb.items():
            try:
                steps = json.loads(key)
                score = float(val[0])
            except Exception:
                continue
            parsed.append((score, steps))
        parsed.sort(key=lambda x: x[0], reverse=True)

        # Tally step-status tags across ALL routes
        tally = _Counter()
        for _score, steps in parsed:
            for step in steps:
                tag = self._step_status_tag(step)
                # Bucket the parametric tag for tally readability
                if tag.startswith("unpurchasable_leaves"):
                    tally["unpurchasable_leaves"] += 1
                else:
                    tally[tag] += 1

        passing = [(s, st) for s, st in parsed if s == 0.0]
        is_pass = bool(passing)
        best_score = parsed[0][0] if parsed else float("-inf")
        n_routes = len(parsed)

        parts = [self._describe_target(target), ""]
        if is_pass:
            shortest = min(passing, key=lambda x: len(x[1]))
            parts.append(
                f"OUTCOME: PASS    pass_steps={len(shortest[1])}    routes_evaluated={n_routes}"
            )
        else:
            parts.append(
                f"OUTCOME: FAIL    best_score={best_score:.3f}    routes_evaluated={n_routes}"
            )
        parts.append("")
        parts.append("FAILURE-MODE TALLY (step-tag occurrences across all routes):")
        for tag in (
            "unpurchasable_leaves",
            "reaction_not_in_USPTO",
            "invalid_template_application",
            "partial_progress",
            "ok_purchasable_leaf",
        ):
            parts.append(f"    {tag:<32s}: {tally.get(tag, 0)}")

        parts.append("")
        parts.append("TOP 3 ATTEMPTED ROUTES (sorted by score, highest first):")
        for i, (score, steps) in enumerate(parsed[:3]):
            parts.append("")
            parts.append(self._render_route(
                steps,
                f"  route #{i+1}   score={score:.3f}   steps={len(steps)}",
            ))

        if is_pass:
            shortest = min(passing, key=lambda x: len(x[1]))
            parts.append("")
            parts.append(self._render_route(
                shortest[1],
                f"  PASSING ROUTE   steps={len(shortest[1])}",
            ))

        summary_text = "\n".join(parts)
        # Best route to attach as a concrete example: the shortest passing route
        # if solved, else the highest-scoring attempted route.
        if is_pass:
            example_steps = min(passing, key=lambda x: len(x[1]))[1]
        else:
            example_steps = parsed[0][1] if parsed else None
        stats = {
            "status": "PASS" if is_pass else "FAIL",
            "best_score": best_score if parsed else None,
            "n_routes": n_routes,
            "best_route_steps": example_steps,
            "failure_counts": {
                k: tally.get(k, 0)
                for k in ("unpurchasable_leaves", "reaction_not_in_USPTO",
                          "invalid_template_application", "partial_progress",
                          "ok_purchasable_leaf")
            },
        }
        return summary_text, stats

    # Strategy-guide size governance: the updater REPLACES, it does not accumulate.
    GUIDE_TARGET_CHARS = 8000   # soft target communicated to the updater LLM
    MAX_GUIDE_CHARS = 10000     # hard cap; oversized output triggers one shrink retry

    _GUIDE_UPDATE_TEMPLATE = (
        "You maintain a retrosynthesis strategy guide read at the start of every new\n"
        "molecule's planning by another LLM. Update the guide based on the latest\n"
        "molecule's run.\n\n"
        "REQUIREMENTS - the guide must be:\n"
        "- HIGH-LEVEL: anti-patterns, heuristics, common failure modes only\n"
        "- GENERALISABLE: NEVER mention specific SMILES strings, atom indices,\n"
        "  reaction template IDs, molecule names, or the specific target just run\n"
        "- CONCISE: target ~{target_chars} characters, HARD LIMIT {max_chars}\n"
        "  characters. The guide REPLACES, it does not accumulate - it must not\n"
        "  grow without bound. For every bullet you add, merge or delete a weaker,\n"
        "  redundant, or evidence-contradicted bullet so the total stays near the\n"
        "  target. Consolidate aggressively; prefer editing existing bullets over\n"
        "  appending.\n"
        "- ACTIONABLE: each lesson is a DO or DON'T the planner can apply\n\n"
        "CURRENT GUIDE:\n"
        "---\n{current_guide}\n---\n\n"
        "LATEST MOLECULE RESULT (reason about WHY routes failed or succeeded, then\n"
        "extract any generalisable lesson):\n"
        "{mol_summary}\n\n"
        "OUTPUT: the full updated markdown guide. No commentary, no preamble, no\n"
        "\"here is the updated guide\" - just the markdown. If the run yields no new\n"
        "generalisable lesson, output the current guide unchanged.\n"
    )

    def _update_strategy_guide(self, target):
        """Acquire fcntl lock, summarise mol, LLM-rewrite the guide, atomic-write back.

        Failures here never abort the molecule loop (caller wraps in try/except).
        See plan Sections 3-6 + 9.
        """
        guide_path = self.llm_config.get('strategy_guide_path', '')
        if not guide_path:
            return
        out_dir = getattr(self.args, 'output_dir', '.') if hasattr(self, 'args') else '.'
        lock_path = guide_path + '.lock'  # alongside guide — shared across runs
        # History + log live under output_dir so each run's audit trail is self-contained
        history_dir = os.path.join(out_dir, os.path.basename(guide_path) + '.history')
        log_path = os.path.join(out_dir, 'guide_update_log.jsonl')
        try:
            os.makedirs(history_dir, exist_ok=True)
        except Exception:
            pass

        summary_text, stats = self._build_mol_summary(target)

        t_lock_start = _time.time()
        # Sidecar lock file — opened separately so we never lock the guide itself
        # (atomic rename re-creates the inode each write).
        with open(lock_path, 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            lock_wait_s = _time.time() - t_lock_start

            current = ""
            if os.path.exists(guide_path):
                try:
                    with open(guide_path) as f:
                        current = f.read()
                except Exception:
                    current = ""

            meta_prompt = self._GUIDE_UPDATE_TEMPLATE.format(
                current_guide=current if current.strip() else "(currently empty - this is the first lesson to seed the guide)",
                mol_summary=summary_text,
                target_chars=self.GUIDE_TARGET_CHARS,
                max_chars=self.MAX_GUIDE_CHARS,
            )

            prior_mode = self.llm_config.get('run_mode', 'A')
            self.llm_config['run_mode'] = 'guide_update'
            t_llm_start = _time.time()
            response = None
            llm_error = None
            new_guide = ""
            outcome = None
            attempts = 0
            try:
                prompt = meta_prompt
                while attempts < 2:
                    attempts += 1
                    try:
                        _msg, response = self.query_LLM(prompt, temperature=0.3)
                    except Exception as e:
                        llm_error = f"{type(e).__name__}: {e}"
                        break
                    new_guide = (response or "").strip()
                    if not new_guide:
                        outcome = "rejected_empty"
                        break
                    if len(new_guide) > self.MAX_GUIDE_CHARS:
                        outcome = "rejected_oversized"
                        # One shrink retry: tell the updater exactly how far over it is.
                        if attempts < 2:
                            prompt = meta_prompt + (
                                f"\n\nYOUR PREVIOUS OUTPUT WAS {len(new_guide)} "
                                f"CHARACTERS - OVER THE {self.MAX_GUIDE_CHARS}-CHARACTER "
                                f"HARD LIMIT. Rewrite it to AT MOST "
                                f"{self.GUIDE_TARGET_CHARS} characters by merging and "
                                f"deleting the weakest, most redundant bullets. Keep "
                                f"the strongest lessons. Output only the markdown guide."
                            )
                            continue
                        break
                    outcome = "no_change" if new_guide == current.strip() else "updated"
                    break
            finally:
                self.llm_config['run_mode'] = prior_mode
            llm_call_s = _time.time() - t_llm_start
            if llm_error is not None:
                outcome = "llm_error"

            md5_before = hashlib.md5(current.encode("utf-8")).hexdigest() if current else ""
            md5_after = md5_before
            if outcome == "updated":
                tmp = guide_path + f".tmp.{os.getpid()}"
                with open(tmp, 'w') as f:
                    f.write(new_guide)
                os.replace(tmp, guide_path)
                md5_after = hashlib.md5(new_guide.encode("utf-8")).hexdigest()
                # Snapshot
                ts_safe = _datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
                snap = os.path.join(
                    history_dir,
                    f"{ts_safe}_pid{os.getpid()}_idx{getattr(self, 'tb_target_idx', -1)}.md",
                )
                try:
                    with open(snap, 'w') as f:
                        f.write(new_guide)
                except Exception:
                    pass

            log_entry = {
                "ts": _datetime.utcnow().isoformat(),
                "pid": os.getpid(),
                "target_idx": getattr(self, 'tb_target_idx', -1),
                "target_smi": target,
                "status": stats["status"],
                "best_score": stats["best_score"],
                "n_routes": stats["n_routes"],
                "failure_counts": stats["failure_counts"],
                "lock_wait_s": round(lock_wait_s, 3),
                "llm_call_s": round(llm_call_s, 3),
                "guide_chars_before": len(current),
                "guide_chars_after": len(new_guide) if outcome == "updated" else len(current),
                "guide_md5_before": md5_before,
                "guide_md5_after": md5_after,
                "outcome": outcome,
                "guide_update_attempts": attempts,
            }
            if llm_error:
                log_entry["error"] = llm_error
            try:
                with _APPEND_LOCK, open(log_path, 'a') as lf:
                    lf.write(json.dumps(log_entry) + "\n")
            except Exception:
                pass

            print(
                f"[guide-update] pid={os.getpid()} idx={getattr(self, 'tb_target_idx', -1)} "
                f"{stats['status']} lock_wait={lock_wait_s:.1f}s llm={llm_call_s:.1f}s "
                f"{len(current)}->{log_entry['guide_chars_after']} chars ({outcome})"
            )
            # Lock released on context-manager exit

    # ------------------------------------------------------------------
    # Evolving skill-structured strategy memory (Run S).
    # See skill_memory.py and the Run S plan. Replaces the flat self-updating
    # guide above with a categorized, statistic-bearing, bandit-sampled store.
    # ------------------------------------------------------------------

    _SKILL_EXTRACT_TEMPLATE = (
        "You are shown ONE retrosynthesis route for a target that SOLVED it or came\n"
        "close to the real synthesis. Your job: extract 1-2 reusable SKILLS that\n"
        "capture the SPECIFIC disconnections this route used, so they can be reused\n"
        "on similar targets. The route worked/approached ground truth, so it WILL\n"
        "contain at least one concrete transferable disconnection - FIND IT.\n\n"
        "You MUST output at least one skill UNLESS the route is a single trivial\n"
        "step on already-purchasable material (only then output []).\n\n"
        "HOW TO PHRASE (this is the important part): name the SPECIFIC reaction and\n"
        "copy the REAL precursor fragments (SMILES) straight from the route. Do NOT\n"
        "generalise into meta-advice - these phrasings are USELESS (the model already\n"
        "believes it follows them) and are NOT acceptable as a skill:\n"
        "  - 'prioritize/use USPTO-validated reactions', 'avoid non-USPTO reactions'\n"
        "  - 'validate/check leaf purchasability', 'avoid unpurchasable fragments'\n"
        "  - 'avoid non-standard / complex / risky steps' or any bare 'avoid X'.\n"
        "Instead of 'use a USPTO amide coupling', write the actual disconnection with\n"
        "real fragments, e.g. 'Ar-C(=O)-NH-R => Ar-C(=O)-OH + H2N-R (amide coupling)'.\n\n"
        "Each skill is a JSON object with these keys:\n"
        "- \"category\": exactly one of: {categories}\n"
        "- \"name\": a short descriptive title naming the REACTION (<=8 words),\n"
        "  e.g. 'Benzylic C-N azide via bromide + NaN3'.\n"
        "- \"trigger\": the structural motif in the TARGET this applies to, e.g.\n"
        "  'primary alkyl azide on a saturated ring', 'aryl-aryl bond between two\n"
        "  heteroarenes', 'tertiary Boc-protected spirocyclic amine'.\n"
        "- \"instructions\": 'IF <specific motif> THEN disconnect <which exact bond>\n"
        "  using <named reaction> to give <the concrete precursor fragments, as\n"
        "  SMILES or clear structures>'. Name the reagent/conditions. MUST reference\n"
        "  real fragments from the routes below - not a schematic.\n"
        "- \"template\": the disconnection as a concrete SMILES/SMARTS transform with\n"
        "  REAL atoms, e.g. 'R-CH2-N3 => R-CH2-Br + NaN3' or\n"
        "  'c1ccc(-c2ccccc2)cc1 => c1ccc(B(O)O)cc1 + Brc1ccccc1 (Suzuki)'. Required.\n"
        "- \"reasoning\": the SPECIFIC reason this disconnection scored well / the\n"
        "  specific reason the failed one failed (e.g. 'the succinimide-N-Br template\n"
        "  is not in USPTO; the real route brominates via ...'). 1-2 sentences.\n\n"
        "RULES:\n"
        "- Anchor every skill on the ACTUAL winning or ground-truth-closest route\n"
        "  below: copy its real precursor SMILES into template/instructions.\n"
        "- Prefer DO-tactics from the highest-similarity routes. A DON'T is allowed\n"
        "  ONLY if it names the specific bad template AND the specific better one.\n"
        "- Two skills must be DIFFERENT reactions, not rephrasings.\n"
        "- Default to extracting a skill; output [] ONLY for a trivial single-step\n"
        "  route on already-purchasable material.\n\n"
        "LATEST MOLECULE RESULT (reason about the SPECIFIC chemistry that worked/failed):\n"
        "{mol_summary}\n\n"
        "OUTPUT: a JSON array (and nothing else) of at most two skill objects.\n"
        "Example: [{{\"category\": \"functional_group_interconversion\",\n"
        "  \"name\": \"Alkyl azide via bromide + NaN3\",\n"
        "  \"trigger\": \"primary/secondary alkyl azide on a saturated carbon\",\n"
        "  \"instructions\": \"IF the target has an alkyl azide R-CH(N3)- on a "
        "saturated ring carbon THEN disconnect C-N3 by SN2 to the bromide R-CH(Br)- "
        "plus NaN3, e.g. COc1cc2c(nc1C)CCCC2N=[N+]=[N-] => COc1cc2c(nc1C)CCCC2Br. "
        "Then make the bromide from the parent CH via a USPTO-attested bromination "
        "(NOT succinimidyl-N-Br, whose template is absent).\",\n"
        "  \"template\": \"R-CH(N3)- => R-CH(Br)- + NaN3\",\n"
        "  \"reasoning\": \"Azide SN2 on a secondary bromide is well attested; the "
        "failure came from brominating with an out-of-corpus NBS template.\"}}]\n"
    )

    # Semantic dedup (true LLM matching, replacing the difflib-only key): decide
    # whether a freshly-extracted skill is the SAME TACTIC as one already stored
    # (just reworded / different numbers) so its example folds into that skill
    # instead of spawning a near-duplicate. Returns an existing id or NONE.
    _SKILL_DEDUP_TEMPLATE = (
        "You curate a retrosynthesis skill memory. Decide whether a NEW skill is\n"
        "the SAME underlying tactic as one already stored, or genuinely distinct.\n"
        "Two skills are the SAME if they prescribe the same disconnection / reaction\n"
        "choice for the same kind of situation, EVEN IF worded differently, citing\n"
        "different example molecules, or different failure counts.\n\n"
        "NEW SKILL:\n"
        "  name: {name}\n"
        "  trigger: {trigger}\n"
        "  instructions: {instructions}\n\n"
        "EXISTING SKILLS (same category):\n{candidates}\n\n"
        "If the NEW skill is the same tactic as ONE existing skill, reply with that\n"
        "skill's id (e.g. s0009). If it is genuinely distinct, reply NONE.\n"
        "Reply with ONLY the id or the word NONE - nothing else."
    )

    # Contextual relevance (port of the paper's B(x_u, m) term): when the store
    # holds more than k_inject skills, an LLM SCORES a UCB-prefiltered candidate
    # pool for applicability to THIS target. The scores feed Thompson sampling
    # (they do not by themselves pick the slate). Falls back to uniform 0.5 on any
    # parse/LLM failure.
    _SKILL_RANK_TEMPLATE = (
        "You score how RELEVANT each retrosynthesis skill is to the target molecule\n"
        "below, judged by how well each skill's situation matches this target's\n"
        "structure.\n\n"
        "TARGET MOLECULE:\n{target_desc}\n\n"
        "CANDIDATE SKILLS (index: [name] (when applies) instructions):\n{candidates}\n\n"
        "OUTPUT: a JSON object (and nothing else) mapping each candidate index (as a\n"
        "string) to a relevance score between 0.0 (irrelevant) and 1.0 (highly\n"
        "applicable). Example: {{\"0\": 0.9, \"1\": 0.1, \"2\": 0.5}}\n"
    )

    def _skill_fingerprints(self, skills):
        """Morgan (ECFP4) fingerprints for each skill's example-product SMILES.
        Cached across molecules (frozen store => compute once)."""
        cache = getattr(self, '_skill_fp_cache', None)
        if cache is not None and cache[0] == len(skills):
            return cache[1]
        fps = []
        for s in skills:
            ex = (s.get("examples") or [{}])[0]
            smi = (ex.get("target_smiles") or "").strip()
            fp = None
            if smi:
                try:
                    m = Chem.MolFromSmiles(smi)
                    if m:
                        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
                except Exception:
                    fp = None
            fps.append(fp)
        self._skill_fp_cache = (len(skills), fps)
        return fps

    def _fg_matcher(self, skills):
        """Build (once, cached) the functional-group applicability matcher over the
        store. Chemistry-semantic: retrieves skills by which reactive groups the
        target carries, aggregated over ALL of each skill's examples -- not by
        whole-molecule scaffold similarity (which buries e.g. azide skills for an
        azide target under scaffold-mates that share only a benzene ring)."""
        cache = getattr(self, '_fg_matcher_cache', None)
        if cache is not None and cache[0] == len(skills):
            return cache[1]
        from skill_matcher import FGMatcher
        m = FGMatcher(skills)
        self._fg_matcher_cache = (len(skills), m)
        return m

    def _tanimoto_prefilter(self, target, skills, pool_n):
        """Stage-1 prefilter: functional-group applicability (see _fg_matcher).
        Falls back to whole-store order if the target or matcher fails. (Name kept
        for call-site stability; the ECFP4 scaffold prefilter it replaced buried
        FG-relevant skills.)"""
        try:
            m = self._fg_matcher(skills)
            ranked, _ = m.rank(target, pool_n)
            return [skills[i] for i, _ in ranked]
        except Exception:
            return list(skills[:pool_n])

    def _rag_installed_fgs(self, smi, k=12):
        """The functional groups RAG's retrieval already SHOWS how to install for
        `smi`: union of installed-FGs (product FGs minus reactant FGs) across the
        reactions in its top-k structurally-similar USPTO routes. Deterministic
        top-k (not the stochastic sample) so it stably characterises RAG coverage.
        Reactions are plain SMILES (no atom maps), so we work at the FG level."""
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
        from skill_matcher import fg_set
        import heapq
        all_fps = getattr(self, '_rag_all_fps', None)
        route_list = getattr(self, '_rag_route_list', None)
        if not all_fps or route_list is None:
            return set()
        # Memoize per fragment: the same intermediate is queried at many
        # modification steps, and each miss is a 299k-route Tanimoto scan (RAG's
        # most expensive op). Without this cache the per-connection arm ran ~7x
        # slower than baseline. Cache is cleared per molecule in optimize().
        cache = getattr(self, '_rag_cov_cache', None)
        if cache is None:
            cache = self._rag_cov_cache = {}
        if smi in cache:
            return cache[smi]
        m = Chem.MolFromSmiles(smi or '')
        if m is None:
            cache[smi] = set()
            return cache[smi]
        fp = AllChem.GetMorganFingerprint(m, 2, useFeatures=False)
        sims = DataStructs.BulkTanimotoSimilarity(fp, list(all_fps))
        covered = set()
        for idx in heapq.nlargest(k, range(len(sims)), key=lambda i: sims[i]):
            for rxn in (route_list[idx] or []):
                try:
                    p, r = rxn.split(">>")
                    covered |= (fg_set(p) - fg_set(r))
                except Exception:
                    pass
        cache[smi] = covered
        return covered

    def _skill_installed_fgs(self, skill):
        """FGs a skill installs, from its best worked example (product − reactants)."""
        from skill_matcher import fg_set
        for e in sorted(skill.get("examples", []) or [],
                        key=lambda e: e.get("gt_similarity", 0.0), reverse=True):
            for st in (e.get("steps") or []):
                if "=>" in st:
                    p, r = st.split("=>", 1)
                    inst = fg_set(p.strip()) - fg_set(r.strip())
                    if inst:
                        return inst
        return set()

    def _sa_score(self, smi):
        """Synthetic-accessibility score (Ertl & Schuffenhauer 2009), cached.
        Higher = harder to make. Precursor SA vs fragment SA measures whether a
        disconnection actually SIMPLIFIES toward buyable building blocks."""
        cache = getattr(self, '_sa_cache', None)
        if cache is None:
            cache = self._sa_cache = {}
        if smi in cache:
            return cache[smi]
        v = 10.0
        try:
            import os, sys
            from rdkit.Chem import RDConfig
            if getattr(self, '_sascorer', None) is None:
                sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
                import sascorer
                self._sascorer = sascorer
            from rdkit import Chem
            m = Chem.MolFromSmiles(smi or '')
            if m is not None:
                v = self._sascorer.calculateScore(m)
        except Exception:
            pass
        cache[smi] = v
        return v

    def _precursor_quality(self, template, frag_smi):
        """rdchiral-APPLY the retro template to the fragment (Coley 2019) and score
        the resulting precursors. Returns (applies, sa_reduction, purch_frac):
          applies    -- the template actually fires (a stricter check than
                        substructure match: many patterns match but produce no
                        valid outcome for THIS substrate).
          sa_reduction -- fragment SA minus mean precursor SA of the best outcome;
                        positive => the cut yields simpler precursors (SCScore/
                        SAScore intuition -- prefer disconnections toward buyables).
          purch_frac -- fraction of the best outcome's precursors already
                        purchasable in inventory (a direct 'reaches building
                        blocks' signal). Cached per (template, fragment)."""
        cache = getattr(self, '_precq_cache', None)
        if cache is None:
            cache = self._precq_cache = {}
        key = (template, frag_smi)
        if key in cache:
            return cache[key]
        applies, sa_red, purch = False, None, 0.0
        try:
            from rdchiral.main import rdchiralRunText
            outcomes = rdchiralRunText(template, frag_smi) or []
            if outcomes:
                applies = True
                fsa = self._sa_score(frag_smi)
                best_red, best_pf = -9.0, 0.0
                for out in outcomes:
                    precs = [p for p in out.split('.') if p]
                    if not precs:
                        continue
                    red = fsa - sum(self._sa_score(p) for p in precs) / len(precs)
                    pf = self._purchasable_frac(precs)
                    # rank outcomes by purchasability then simplicity
                    if (pf, red) > (best_pf, best_red):
                        best_pf, best_red = pf, red
                sa_red, purch = best_red, best_pf
        except Exception:
            pass
        cache[key] = (applies, sa_red, purch)
        return cache[key]

    def _purchasable_frac(self, smi_list):
        """Fraction of molecules purchasable in inventory (reuses the oracle's own
        building-block set). 0 if inventory unavailable."""
        inv = getattr(self, 'inventory', None)
        if inv is None or not smi_list:
            return 0.0
        try:
            from modification_prompt import Molecule
        except Exception:
            return 0.0
        ok = 0
        for smi in smi_list:
            try:
                if inv.is_purchasable(Molecule(smi)):
                    ok += 1
            except Exception:
                pass
        return ok / len(smi_list)

    def _is_purchasable(self, smi):
        """Is this fragment a purchasable building block? The stop signal for
        recursion: once a child is buyable, the sub-problem is solved. Cached."""
        cache = getattr(self, '_purch_cache', None)
        if cache is None:
            cache = self._purch_cache = {}
        if smi in cache:
            return cache[smi]
        inv = getattr(self, 'inventory', None)
        v = False
        if inv is not None:
            try:
                from modification_prompt import Molecule
                v = bool(inv.is_purchasable(Molecule(smi)))
            except Exception:
                v = False
        cache[smi] = v
        return v

    def _apply_best_outcome(self, template, frag):
        """rdchiral-apply the template to `frag` and return the precursor SMILES of
        the BEST outcome (most purchasable, then simplest). None if it doesn't fire.
        This is the look-ahead: the actual children this disconnection produces, so
        the guided plan can recurse on them and mark which are already buyable."""
        try:
            from rdchiral.main import rdchiralRunText
            outs = rdchiralRunText(template, frag) or []
        except Exception:
            return None
        best, bkey = None, (-1.0, -9.0)
        fsa = self._sa_score(frag)
        for out in outs:
            precs = [p for p in out.split('.') if p]
            if not precs:
                continue
            pf = self._purchasable_frac(precs)
            sa = fsa - sum(self._sa_score(p) for p in precs) / len(precs)
            if (pf, sa) > bkey:
                bkey, best = (pf, sa), precs
        return best

    def _rank_valid_skills(self, frag_smi, frag_mol, skills):
        """Robust ranking of retron-valid skills for one fragment, chemistry-only
        (the frozen USPTO store carries no usable trials/gt signal -- all trials=0,
        gt hardcoded 1.0). Composite of four literature-grounded signals:
          consensus  -- fraction of valid skills sharing this disconnection
                        signature (template-popularity prior, RetroSim 2017);
          specificity-- heavy atoms the product pattern constrains (less
                        coincidental spectator matches);
          fg         -- functional-group applicability (topical relevance);
          precursor  -- rdchiral-apply + purchasable-fraction + SA-reduction
                        (does the cut fire AND simplify toward buyables?).
        Two-stage for cost: cheap (consensus+spec+fg) over ALL valid, then
        rdchiral/SA/purchasability only on the cheap top-K."""
        from skill_matcher import retron_matches, retron_specificity, fg_set
        from collections import Counter
        retron_cache = self._retron_cache
        if retron_cache is None:
            retron_cache = self._retron_cache = {}
        # retron-valid set (substructure gate; whole store)
        valid = []
        for s in skills:
            if not (s.get("reasoning") or "").strip():
                continue
            tmpl = s.get("template") or ""
            ck = (id(s), frag_smi)
            ok = retron_cache.get(ck)
            if ok is None:
                ok = retron_matches(frag_mol, tmpl) if frag_mol is not None else False
                retron_cache[ck] = ok
            if ok:
                valid.append(s)
        if not valid:
            return []
        # consensus: signature popularity among valid skills
        sigc = Counter()
        for s in valid:
            sg = s.get("signature") or ""
            if sg:
                sigc[sg] += 1
        nv = float(len(valid))
        # fg scores aligned to the whole store, looked up by id
        try:
            fgm = self._fg_matcher(skills)
            fg_scores = fgm.score(fg_set(frag_smi))
            fg_by_id = {id(s): fg_scores[i] for i, s in enumerate(skills)}
        except Exception:
            fg_by_id = {}
        def cheap(s):
            cons = sigc.get(s.get("signature") or "", 0) / nv
            spec = min(retron_specificity(frag_mol, s.get("template") or ""), 12) / 12.0
            fg = fg_by_id.get(id(s), 0.0)
            return 0.45 * cons + 0.30 * spec + 0.25 * fg
        valid.sort(key=cheap, reverse=True)
        K = int(self._skill_cfg.get('perconn_confirm', 12))
        scored = []
        for s in valid[:K]:
            applies, sa_red, purch = self._precursor_quality(s.get("template") or "", frag_smi)
            if not applies:            # rdchiral-confirm: drop patterns that match but don't fire
                continue
            simp = max(-1.0, min(1.0, (sa_red or 0.0) / 2.0)) * 0.5 + 0.5   # -> [0,1]
            final = cheap(s) + 0.35 * purch + 0.20 * simp
            scored.append((final, s))
        # Selection MARGIN (top1 - top2), stashed for the abstention gate in
        # _guided_decomposition_plan. Relevance is abundant (~78 highly-applicable
        # skills per target) but only one disconnection is correct, so a FLAT
        # ranking means "many equally-plausible cuts" -> injecting picks one at
        # near-random and steers the population off the productive path. A wide
        # margin means the composite actually discriminates. 0.0 when there is
        # only one candidate (nothing to be ambiguous with -> never abstain).
        # any skill that survived rdchiral-confirm outranks those we didn't confirm
        if not scored:                 # nothing fired in top-K -> fall back to cheap order
            cs = sorted((cheap(s) for s in valid), reverse=True)
            self._last_rank_margin = (cs[0] - cs[1]) if len(cs) > 1 else 1.0
            return valid
        scored.sort(key=lambda t: t[0], reverse=True)
        self._last_rank_margin = (scored[0][0] - scored[1][0]) if len(scored) > 1 else 1.0
        return [s for _, s in scored] + [s for s in valid[:K] if s not in [x for _, x in scored]]

    # NOTE on prose. The retron gate guarantees the TEMPLATE fires, but
    # `name`/`reasoning` were written for the skill's ORIGIN reaction and travel
    # unchanged onto any fragment the template happens to match. Observed live on
    # the oxadiazolone loss: fragment Cc1noc(=O)n1C/C=C\\CC(=O)O was labelled
    # "N-Alkylation of 2H-3,1-benzoxazin-2-one", reasoning about "the
    # benzoxazinone nitrogen" and "the isobutyl methylene" -- neither present.
    # The template is legitimately applicable (a generic N-alkylation pattern
    # matches both ring systems); only the prose is substrate-specific.
    # A whole-molecule FG-overlap detector was tried and REJECTED: ubiquitous
    # groups (fr_C_O, fr_NH0, ether) make it fire on 3-4% of skills, far too weak
    # to test the hypothesis. `render_reasoning: false` instead ABLATES the prose
    # outright, leaving the strategy name plus the machine-generated (always
    # correct) `=>` transformation. That is a decisive test rather than a
    # heuristic: if losses soften, mis-scaffolded prose was the cause.

    def _violates_monotonicity(self, template, fragment, parent_heavy, tol=1.0):
        """True if applying `template` to `fragment` yields an UNPURCHASABLE child
        with more heavy atoms than the fragment itself (times `tol`).

        Purchasable children are exempt -- a real retro step often produces a large
        buyable partner. False when the template does not fire (that case is
        already handled by the rdchiral confirmation in _rank_valid_skills), so
        this gate only ever rejects a cut that demonstrably grows the problem.
        """
        try:
            from rdkit import Chem
            precs = self._apply_best_outcome(template, fragment)
            if not precs:
                return False
            limit = parent_heavy * tol
            for p in precs:
                pm = Chem.MolFromSmiles(p)
                if pm is None:
                    continue
                if pm.GetNumHeavyAtoms() > limit and not self._is_purchasable(p):
                    return True
            return False
        except Exception:
            return False               # never let the gate itself drop guidance

    def _guided_decomposition_plan(self, fragment, skills, used_sig, used_ids,
                                   depth, max_depth, budget):
        """Recursively guide the sub-problem 'route `fragment` to purchasable
        building blocks'. Picks the composite-best retron-valid skill for the
        fragment, rdchiral-applies it to see the actual children, marks each child
        PURCHASABLE (stop) or recurses on it up to max_depth. This is what makes
        DEEP nodes individually guided: the fragments the model would otherwise
        invent mid-response (and never get a skill for) are pre-expanded here, each
        with its own best cut + a buyable stop-signal. Returns rendered lines."""
        from rdkit import Chem
        if budget[0] <= 0 or not fragment:
            return []
        frag_mol = Chem.MolFromSmiles(fragment)
        if frag_mol is None:
            return []
        chosen = None
        self._last_rank_margin = 1.0
        ranked = self._rank_valid_skills(fragment, frag_mol, skills)
        # ABSTENTION GATE: if the ranking could not separate its top two
        # candidates, decline to guide this node rather than inject an
        # arbitrary pick. 0.0 (default) keeps the always-inject behaviour of
        # every prior arm.
        margin_min = float(self._skill_cfg.get('abstain_margin', 0.0))
        mg = getattr(self, '_last_rank_margin', 1.0)
        # Log every margin so the threshold can be calibrated FROM THIS RUN
        # (the distribution is not obtainable offline -- scoring needs the
        # inventory, which OOMs on the login node).
        if self._skill_cfg.get('log_margin', False):
            print("[margin] depth=%d n_ranked=%d margin=%.4f thr=%.4f %s"
                  % (depth, len(ranked), mg, margin_min,
                     "ABSTAIN" if (margin_min > 0.0 and mg < margin_min) else "inject"),
                  flush=True)
        if margin_min > 0.0 and mg < margin_min:
            self._abstained = getattr(self, '_abstained', 0) + 1
            return []
        # MONOTONICITY GATE: a retro step must not hand back an UNPURCHASABLE
        # child bigger than the fragment it replaced. Diagnosed on the
        # oxadiazolone loss, where the greedy tree walked
        #   alcohol -> enone -> alpha-bromo enone -> alpha-bromo allylic IODIDE
        # i.e. it added halogens and grew the molecule while claiming to head
        # toward building blocks, and every leaf ended [decompose further].
        # Purchasable children are exempt: a legitimate retro step often yields a
        # LARGE buyable partner (Boc2O, a boronic acid, a coupling reagent) and
        # penalising those would reject good chemistry. Skills that violate are
        # skipped in rank order, so a compliant lower-ranked cut is used instead
        # of abandoning the node. Bounded by monotonic_scan to cap rdchiral cost.
        mono = bool(self._skill_cfg.get('monotonic_gate', False))
        mono_tol = float(self._skill_cfg.get('monotonic_tol', 1.0))
        mono_scan = int(self._skill_cfg.get('monotonic_scan', 5))
        parent_heavy = frag_mol.GetNumHeavyAtoms()
        tried = 0
        for s in ranked:
            sig = self._sig.dedup_key(s)
            if sig and sig in used_sig:
                continue
            if mono and tried < mono_scan:
                tried += 1
                if self._violates_monotonicity(s.get("template") or "", fragment,
                                               parent_heavy, mono_tol):
                    self._mono_rejected = getattr(self, '_mono_rejected', 0) + 1
                    continue
            chosen = (s, sig)
            break
        if chosen is None:
            return []
        s, sig = chosen
        if sig:
            used_sig.add(sig)
        used_ids.append(s.get("id"))
        budget[0] -= 1
        ind = "  " * depth
        name = (s.get('name') or '').strip()
        lines = [f"{ind}- sub-problem: {fragment}",
                 f"{ind}    strategy: {name}"]
        # Reasoning only at the TOP node, truncated to its first two sentences (the
        # transferable mechanistic core -- "cleaves the C-N bond via SN2..."). The
        # later sentences and all deeper-node prose reference the skill's ORIGINAL
        # example molecule ("the difluorophenyl side chain"), which is noise for the
        # fragment at hand -- so deeper nodes render only strategy + transformation.
        if depth == 0:
            rz = (s.get('reasoning') or '').strip()
            if rz and self._skill_cfg.get('render_reasoning', True):
                sents = rz.split('. ')
                lines.append(f"{ind}    reasoning: {'. '.join(sents[:2]).rstrip('.') + '.'}")
        precs = self._apply_best_outcome(s.get("template") or "", fragment)
        if precs:
            lines.append(f"{ind}    => {' . '.join(precs)}")
            min_atoms = int(self._skill_cfg.get('min_frag_atoms', 6))
            # Recurse on every child that is a REAL intermediate: not purchasable
            # (inventory distinguishes reagents/coupling-partners -- which ARE
            # buyable -> stop -- from genuine fragments that need synthesis) and not
            # reagent-sized. So both fragments of a convergent cut get guided, while
            # NBS/Boc2O/CCCl are pruned by the buyability check. (The recurse-into-
            # reagents garbage only appears WITHOUT inventory, e.g. in offline tests.)
            for p in precs:
                pm = Chem.MolFromSmiles(p)
                na = pm.GetNumHeavyAtoms() if pm is not None else 0
                if self._is_purchasable(p):
                    lines.append(f"{ind}      {p}  [PURCHASABLE — stop]")
                elif na < min_atoms:
                    lines.append(f"{ind}      {p}  [reagent/building block — stop]")
                elif depth + 1 < max_depth and budget[0] > 0:
                    child = self._guided_decomposition_plan(
                        p, skills, used_sig, used_ids, depth + 1, max_depth, budget)
                    lines.extend(child if child else [f"{ind}      {p}  [decompose further]"])
                else:
                    lines.append(f"{ind}      {p}  [decompose further]")
        return lines

    def _select_skills_per_connection(self, intermediates):
        """RECURSIVE LOOK-AHEAD skill injection. For each unpurchasable intermediate,
        build a guided DECOMPOSITION TREE (not a single cut): the composite-best
        retron-valid disconnection, its rdchiral-predicted children, and -- for any
        child still unpurchasable -- ITS best disconnection, recursively to
        max_depth. Purchasable children are marked as stop points.

        Why a tree, not one cut (diagnosis of the oxadiazolone loss): the model
        solves several sub-problems in ONE modification response, so guiding only
        the route's current leaf left the DEEP fragments (that it then invents)
        unguided -- and those are where it dead-ended. Pre-expanding the tree gives
        every anticipated node its own best cut + a buyability stop-signal in the
        same prompt, and the stop-signal also curbs the over-decomposition failure.
        Reuses the robust composite (retron gate + consensus/specificity/FG +
        rdchiral-apply/SA/purchasability) at every node."""
        if not self._skill_memory_enabled or not self._skill_memory:
            return ""
        skills = [s for s in self._skill_memory.get("skills", [])
                  if s.get("state") != self._sm.STATE_ELIMINATED]
        if not skills:
            return ""
        if getattr(self, '_retron_cache', None) is None:
            self._retron_cache = {}
        max_depth = int(self._skill_cfg.get('perconn_depth', 2))
        budget = [int(self._skill_cfg.get('perconn_max_nodes', 10))]
        used_sig, used_ids, all_lines = set(), [], []
        for smi in (intermediates or []):
            if not smi or self._is_purchasable(smi):
                continue                       # already buyable -> nothing to solve
            plan = self._guided_decomposition_plan(
                smi, skills, used_sig, used_ids, 0, max_depth, budget)
            if plan:
                all_lines.extend(plan)
            if budget[0] <= 0:
                break
        if not all_lines:
            return ""
        header = ["\n=== GUIDED DECOMPOSITION — route each fragment to purchasable "
                  "building blocks (deeper indent = one disconnection further; "
                  "[PURCHASABLE — stop] = do not decompose further) ==="]
        header.extend(all_lines)
        header.append("=== END GUIDED DECOMPOSITION ===\n")
        self._injected_skill_ids = used_ids
        if not getattr(self, '_injected_skill_ids_all', None):
            self._injected_skill_ids_all = set()
        self._injected_skill_ids_all.update(used_ids)
        return "\n".join(header)

    def _select_skills_for_injection(self, target=None, resample=False):
        """Select skills for the current prompt and return the rendered block.

        Pipeline (Option C):
          1. Tanimoto (ECFP4) PREFILTER the whole store to prefilter_pool candidates
             most similar to the target -- cheap, no LLM, works on 1000s of skills.
          2. LLM-RANK that pool for relevance (one call, feasible pool size).
          3. Inject relevance-sorted skills up to a TOKEN BUDGET (skill_token_budget)
             and above a relevance_floor -- "as broad as relevant" without a fixed k.
        Relevance/pool are cached per molecule (reset in optimize()); resample=True
        reuses the cache (no extra LLM call). Injected ids union into
        self._injected_skill_ids_all for credit."""
        if not self._skill_memory_enabled:
            return ""
        if not resample and self._injected_skill_block is not None:
            return self._injected_skill_block
        max_ex = int(self._skill_cfg.get('max_examples_render', 1))
        prefilter_pool = int(self._skill_cfg.get('prefilter_pool', 250))
        token_budget = int(self._skill_cfg.get('skill_token_budget', 25000))
        rel_floor = float(self._skill_cfg.get('relevance_floor', 0.35))
        max_inject = int(self._skill_cfg.get('max_inject', 0))   # 0 = no count cap
        fb_k = int(self._skill_cfg.get('fallback_k', 25))
        mem = self._skill_memory or {}
        skills = [s for s in mem.get("skills", [])
                  if s.get("state") != self._sm.STATE_ELIMINATED]
        if not skills:
            self._injected_skill_ids = []
            if not resample:
                self._injected_skill_block = ""
            return ""

        # Force-inject (ceiling test): if a molecule->skill_ids map is configured,
        # inject exactly those oracle-selected skills for the current target,
        # bypassing FG-matcher + LLM ranker. Isolates injection cost from selection.
        fmap_path = self._skill_cfg.get('force_inject_map')
        if fmap_path and target:
            fmap = getattr(self, '_force_map_cache', None)
            if fmap is None:
                try:
                    with open(fmap_path) as fh:
                        fmap = json.load(fh)
                except Exception:
                    fmap = {}
                self._force_map_cache = fmap
            tk = None
            try:
                tm = Chem.MolFromSmiles(target)
                tk = Chem.MolToInchiKey(tm).split('-')[0] if tm else None
            except Exception:
                tk = None
            ids = fmap.get(tk) if tk else None
            if ids:
                idset = set(ids)
                forced = [s for s in skills if s.get("id") in idset]
                if forced:
                    self._injected_skill_ids = [s.get("id") for s in forced]
                    if not getattr(self, '_injected_skill_ids_all', None):
                        self._injected_skill_ids_all = set()
                    self._injected_skill_ids_all.update(self._injected_skill_ids)
                    block = self._sm.render_skills(forced, max_examples=max_ex)
                    if not resample:
                        self._injected_skill_block = block
                    return block

        # 1+2: FG prefilter -> LLM relevance rank (cached per molecule).
        # Decouple recall depth from ranker load: FG-prefilter deep (prefilter_pool),
        # but only LLM-rank the top rank_candidate_pool by FG score -- the ranker
        # degrades and bloats when handed hundreds of candidates at once.
        rank_pool_n = int(self._skill_cfg.get('rank_candidate_pool', 150))
        if getattr(self, '_relevance_cache', None) is None:
            pool = (self._tanimoto_prefilter(target, skills, prefilter_pool)
                    if target else list(skills[:prefilter_pool]))
            pool = pool[:rank_pool_n]   # ranker + injection draw from this slice
            relevance = {}
            if bool(self._skill_cfg.get('contextual_rank', True)) and target and pool:
                try:
                    relevance = self._rank_skills_llm(target, pool) or {}
                except Exception:
                    relevance = {}
            self._relevance_cache = relevance
            self._pool_cache = {"skills": pool}
        relevance = self._relevance_cache
        pool = self._pool_cache.get("skills", [])

        # 3: token-budget injection, relevance-sorted, above the floor.
        ranked = sorted(pool, key=lambda s: relevance.get(s.get("id"), 0.0), reverse=True)

        # DIVERSITY GATE (2026-07-20) -- applied to the RANKED POOL, before the
        # max_inject cap. Order matters: capping first and deduping after would
        # only shrink the block (5 near-duplicates -> 4 skills) instead of
        # reaching deeper for a 5th DISTINCT idea, which is the entire point.
        # On 2026-07-19 all 5 slots went to paraphrases of ONE disconnection
        # (5x ester, 5x carbamate) and the model received one idea five times.
        # Ranking order is preserved, so relevance still decides WHICH skill
        # represents each transformation.
        if bool(self._skill_cfg.get('diversify_injection', True)):
            try:
                _above = [s for s in ranked
                          if relevance.get(s.get("id"), 0.0) >= rel_floor] or ranked
                _div = self._sig.diversify(
                    _above, max_inject or fb_k,
                    per_sig=int(self._skill_cfg.get('diversity_per_sig', 1)),
                    backfill=bool(self._skill_cfg.get('diversity_backfill', False)))
                if _div:
                    print(f'[diversity] {len(_above)} relevant -> {len(_div)} distinct '
                          f'transformations')
                    ranked = _div
            except Exception as e:
                print(f'[diversity] skipped ({e})')

        budget_chars = token_budget * 4    # ~4 chars/token
        selected, chars = [], 0
        for s in ranked:
            if relevance.get(s.get("id"), 0.0) < rel_floor:
                break
            if max_inject and len(selected) >= max_inject:
                break
            slen = sum(len(x) for x in self._sm._render_one(s, max_ex))
            if selected and chars + slen > budget_chars:
                break
            selected.append(s)
            chars += slen
        if not selected:
            # No relevance signal (LLM failed) or nothing cleared the floor ->
            # fall back to the top Tanimoto-similar skills (still on-topic, not
            # arbitrary), bounded by fallback_k (or max_inject if smaller).
            # Fallback path: no relevance signal at all. Diversify here too, so a
            # ranker failure cannot silently reintroduce the paraphrase block.
            selected = list(pool[:(max_inject or fb_k)])
            if bool(self._skill_cfg.get('diversify_injection', True)):
                try:
                    selected = self._sig.diversify(
                        pool, max_inject or fb_k,
                        per_sig=int(self._skill_cfg.get('diversity_per_sig', 1)),
                        backfill=bool(self._skill_cfg.get('diversity_backfill', False)))
                except Exception as e:
                    print(f'[diversity] fallback skipped ({e})')

        ids = [s.get("id") for s in selected]
        self._injected_skill_ids = ids
        if not getattr(self, '_injected_skill_ids_all', None):
            self._injected_skill_ids_all = set()
        self._injected_skill_ids_all.update(ids)
        block = self._sm.render_skills(selected, max_examples=max_ex)
        if not resample:
            self._injected_skill_block = block
        return block

    def _rank_skills_llm(self, target, candidates):
        """Ask the LLM to score `candidates` for relevance to `target`. Returns a
        {skill_id: relevance in [0,1]} map, or {} on any failure."""
        if not candidates:
            return {}
        try:
            target_desc = self._describe_target(target)
        except Exception:
            target_desc = str(target)
        lines = []
        for i, s in enumerate(candidates):
            trig = (s.get("trigger", "") or "any target").strip()
            name = (s.get("name", "") or "").strip()
            instr = (s.get("instructions", "") or s.get("lesson", "")).strip()
            lines.append(f"{i}: [{name}] (when: {trig}) {instr}")
        prompt = self._SKILL_RANK_TEMPLATE.format(
            target_desc=target_desc, candidates="\n".join(lines)
        )
        prior_mode = self.llm_config.get('run_mode', 'A')
        self.llm_config['run_mode'] = 'skill_rank'
        try:
            _msg, response = self.query_LLM(prompt, temperature=0.0)
        finally:
            self.llm_config['run_mode'] = prior_mode
        scores = self._parse_relevance_scores(response or "", len(candidates))
        return {candidates[i].get("id"): v for i, v in scores.items()}

    @staticmethod
    def _parse_relevance_scores(text, n):
        """Parse a JSON object mapping candidate index -> relevance [0,1]. Tolerant
        of fences/prose. Returns {int_index: float} for valid in-range entries."""
        if not text:
            return {}
        s = text.strip()
        if s.startswith("```"):
            s = s.split("\n", 1)[-1]
            if s.endswith("```"):
                s = s[:-3]
        lo, hi = s.find("{"), s.rfind("}")
        if lo == -1 or hi == -1 or hi <= lo:
            return {}
        try:
            obj = json.loads(s[lo:hi + 1])
        except Exception:
            return {}
        out = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                try:
                    i = int(k)
                    f = float(v)
                except (TypeError, ValueError):
                    continue
                if 0 <= i < n:
                    out[i] = max(0.0, min(1.0, f))
        return out

    def _semantic_dedup_match(self, new_sk, candidates):
        """True LLM semantic dedup: is `new_sk` (freshly extracted) the same tactic
        as one of `candidates` (existing skill dicts, same category)? Returns the
        matched existing skill id, or None (empty candidates / NONE reply / any
        LLM or parse failure -> caller falls back to difflib merge_or_add)."""
        if not candidates:
            return None
        lines, valid_ids = [], []
        for s in candidates:
            sid = s.get("id")
            if not sid:
                continue
            valid_ids.append(sid)
            nm = (s.get("name", "") or "").strip()
            tr = (s.get("trigger", "") or "any target").strip()
            ins = (s.get("instructions", "") or s.get("lesson", "")).strip()
            lines.append(f"  {sid}: [{nm}] (when: {tr}) {ins}")
        if not valid_ids:
            return None
        prompt = self._SKILL_DEDUP_TEMPLATE.format(
            name=(new_sk.get("name", "") or "").strip(),
            trigger=(new_sk.get("trigger", "") or "").strip(),
            instructions=(new_sk.get("instructions", "") or "").strip(),
            candidates="\n".join(lines),
        )
        prior_mode = self.llm_config.get('run_mode', 'A')
        self.llm_config['run_mode'] = 'skill_dedup'
        try:
            _msg, response = self.query_LLM(prompt, temperature=0.0)
        except Exception:
            return None
        finally:
            self.llm_config['run_mode'] = prior_mode
        return self._parse_dedup_id(response or "", set(valid_ids))

    @staticmethod
    def _parse_dedup_id(text, valid_ids):
        """Return the first skill id token in `text` that is one of the offered
        `valid_ids`, else None (covers a 'NONE' reply, empty text, or a
        hallucinated id that wasn't a candidate)."""
        if not text:
            return None
        for m in re.findall(r"s\d{3,}", text):
            if m in valid_ids:
                return m
        return None

    @staticmethod
    def _parse_extracted_skills(text):
        """Parse the extractor LLM's JSON array of rich skill objects. Tolerant of
        code fences and surrounding prose. Returns a list of dicts with keys
        category/name/trigger/instructions/template/reasoning."""
        if not text:
            return []
        s = text.strip()
        # Strip markdown code fences if present.
        if s.startswith("```"):
            s = s.split("\n", 1)[-1]
            if s.endswith("```"):
                s = s[:-3]
        # Isolate the outermost JSON array.
        lo, hi = s.find("["), s.rfind("]")
        if lo == -1 or hi == -1 or hi <= lo:
            return []
        try:
            arr = json.loads(s[lo:hi + 1])
        except Exception:
            return []
        out = []
        for item in arr if isinstance(arr, list) else []:
            if isinstance(item, dict):
                # Accept legacy 'lesson' as a fallback for 'instructions'.
                instructions = str(item.get("instructions",
                                            item.get("lesson", ""))).strip()
                if not instructions:
                    continue
                out.append({
                    "category": str(item.get("category", "general")).strip(),
                    "name": str(item.get("name", "")).strip(),
                    "trigger": str(item.get("trigger", "")).strip(),
                    "instructions": instructions,
                    "template": str(item.get("template", "")).strip(),
                    "reasoning": str(item.get("reasoning", "")).strip(),
                })
        return out[:2]  # hard cap: at most two new skills per molecule

    def _update_skill_memory(self, target, attempt=1, ground_truth=None,
                             is_final=True):
        """Credit-assign injected skills (weighted by ground-truth closeness),
        LLM-extract new hierarchical skills with a concrete worked example, evolve
        the lifecycle (4 states) + storage cleanup, persist under fcntl lock.
        Called once per retry attempt; never aborts the molecule loop.
        `total_molecules` is bumped only on the molecule's final attempt so it
        counts molecules, not attempts."""
        path = self._skill_memory_path
        if not path:
            return
        out_dir = getattr(self.args, 'output_dir', '.') if hasattr(self, 'args') else '.'
        lock_p = self._sm.lock_path(path)
        history_dir = os.path.join(out_dir, os.path.basename(path) + '.history')
        log_path = os.path.join(out_dir, 'skill_memory_log.jsonl')

        summary_text, stats = self._build_mol_summary(target)
        if stats["status"] == "PASS":
            reward = 1.0
        else:
            # Oracle score convention: 0.0 = perfect (PASS), more negative = worse;
            # best_score = max(route scores) in (-inf, 0]. Dense partial reward is
            # higher the closer the best route got to solving. Clamp to [0, 1] so
            # winrates stay in [0, 1] and the lifecycle thresholds remain meaningful.
            best = stats["best_score"]
            reward = max(0.0, min(1.0, 1.0 + (best if best is not None else -1.0)))

        idx = getattr(self, 'tb_target_idx', -1)

        # Gated per-modification-route extraction. oracle.route_buffer holds EVERY
        # route the GA explored (each modification's output); _gt_rank_routes ranks
        # that whole trajectory by ground-truth closeness. We extract a skill from
        # EACH trajectory route that clears the GT gate (gt_sim >=
        # extract_gt_threshold) OR that PASSED - learning specific disconnections
        # from the modifications that actually approached the real synthesis, and
        # writing NOTHING from generic far-from-gt failures (the signal that
        # produced the old generic monoculture). This is "extract per modification,
        # gated by GT" evaluated at end-of-molecule over the captured trajectory
        # (no per-step LLM cost during search). Each skill carries its OWN route's
        # example + origin_gt.
        thr = float(self._skill_cfg.get('extract_gt_threshold', 0.5))
        max_routes = max(1, int(self._skill_cfg.get('max_extract_routes', 3)))
        anchor_steps = stats.get("best_route_steps")
        anchor_outcome = stats["status"]
        gt_sim = 0.0
        qualifying = []   # (steps, gt_sim, outcome) routes that clear the gate
        if ground_truth:
            gt_routes = self._gt_rank_routes(
                ground_truth, top_n=max(max_routes,
                                        int(self._skill_cfg.get('gt_rank_top_n', 3))))
            if gt_routes:
                anchor_steps, gt_sim, anchor_outcome, _osc = gt_routes[0]
            for steps, gsim, outc, osc in gt_routes:
                if gsim >= thr or outc == "PASS":
                    qualifying.append((steps, gsim, outc))
                if len(qualifying) >= max_routes:
                    break
        # A PASS always earns a skill even without ground truth / gate.
        if not qualifying and anchor_outcome == "PASS":
            qualifying = [(anchor_steps, gt_sim, "PASS")]
        # Molecule-level example/gt (apply_reward credit + fallback).
        example = self._build_example(anchor_steps, target, anchor_outcome, gt_sim, idx)

        # One extraction LLM call per qualifying route (OUTSIDE the lock). Each
        # returned skill carries its own route's example via '_example'.
        prior_mode = self.llm_config.get('run_mode', 'A')
        self.llm_config['run_mode'] = 'skill_update'
        extracted = []
        llm_error = None
        t_llm = _time.time()
        try:
            for steps, gsim, outc in qualifying:
                rsummary = (summary_text +
                            f"\n\nEXTRACT FROM THIS SPECIFIC ROUTE (gt_similarity="
                            f"{gsim:.2f} to the known synthesis, outcome={outc}):\n"
                            + self._render_route(steps, "  route:"))
                prompt = self._SKILL_EXTRACT_TEMPLATE.format(
                    categories=", ".join(self._sm.FIXED_CATEGORIES),
                    mol_summary=rsummary,
                )
                try:
                    _msg, response = self.query_LLM(prompt, temperature=0.3)
                    for sk in self._parse_extracted_skills(response or ""):
                        sk['_example'] = self._build_example(steps, target, outc, gsim, idx)
                        extracted.append(sk)
                except Exception as e:
                    llm_error = f"{type(e).__name__}: {e}"
        finally:
            self.llm_config['run_mode'] = prior_mode
        llm_call_s = _time.time() - t_llm

        # True semantic dedup (LLM): map each freshly-extracted skill to an
        # existing skill it merely rephrases, so its example folds in instead of
        # spawning a near-duplicate. Done BEFORE the lock (the LLM call is slow;
        # holding the fcntl lock across it would serialise all workers). Matched
        # against a pre-lock snapshot; applied under the lock via merge_into, which
        # re-checks the id and falls back to difflib merge_or_add if the store
        # changed (staleness-safe).
        dedup_s = 0.0
        if extracted and bool(self._skill_cfg.get('semantic_dedup', False)):
            t_dd = _time.time()
            snap = self._sm.load_memory(path)
            for sk in extracted:
                cat = sk.get("category") if sk.get("category") in self._sm.FIXED_CATEGORIES else "general"
                cands = [s for s in snap.get("skills", [])
                         if s.get("category") == cat
                         and s.get("state") != self._sm.STATE_ELIMINATED]
                try:
                    sk["_merge_into"] = self._semantic_dedup_match(sk, cands)
                except Exception:
                    sk["_merge_into"] = None
            dedup_s = _time.time() - t_dd

        t_lock = _time.time()
        with open(lock_p, 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            lock_wait_s = _time.time() - t_lock
            mem = self._sm.load_memory(path)  # reload latest under lock

            n_before = len(mem.get("skills", []))
            # Credit every skill injected anywhere this molecule (init + all
            # modification steps), not just the last draw.
            credit_ids = sorted(getattr(self, '_injected_skill_ids_all', None)
                                or self._injected_skill_ids)
            n_credited = self._sm.apply_reward(
                mem, credit_ids, reward, gt_sim=gt_sim,
                idx=idx, attempt=attempt)

            added = []
            n_merged = 0
            for sk in extracted:
                # merge_into with a None id falls straight through to difflib
                # merge_or_add, so this one call covers both the semantic-match
                # and the no-match / dedup-off paths.
                action, skill = self._sm.merge_into(
                    mem, sk.get("_merge_into"), sk["category"], sk["instructions"], idx,
                    name=sk.get("name", ""), trigger=sk.get("trigger", ""),
                    template=sk.get("template", ""), reasoning=sk.get("reasoning", ""),
                    example=sk.get("_example", example), attempt=attempt,
                )
                if action == "added":
                    added.append(skill.get("id"))
                elif action == "merged":
                    n_merged += 1

            # Lifecycle evolution (paper's 4 states): soft transitions incl.
            # retired->eliminated, then SEPARATE storage cleanup (no-op while
            # max_total_skills<=0 -> unbounded store).
            transitions = self._sm.evolve_lifecycle(
                mem,
                promote_min_trials=int(self._skill_cfg.get('promote_min_trials', 3)),
                promote_winrate=float(self._skill_cfg.get('promote_winrate', 0.5)),
                retire_min_trials=int(self._skill_cfg.get('retire_min_trials', 4)),
                retire_winrate=float(self._skill_cfg.get('retire_winrate', 0.25)),
                eliminate_min_trials=int(self._skill_cfg.get('eliminate_min_trials', 6)),
                eliminate_winrate=float(self._skill_cfg.get('eliminate_winrate', 0.05)),
            )
            removed = self._sm.storage_cleanup(
                mem, max_total=int(self._skill_cfg.get('max_total_skills', 0)),
            )
            # Count molecules, not attempts: bump only on the final attempt.
            if is_final:
                mem["total_molecules"] = mem.get("total_molecules", 0) + 1
            self._sm.atomic_write_memory(mem=mem, path=path, history_dir=history_dir,
                                         tag=f"idx{idx}_a{attempt}")
            self._skill_memory = mem
            n_after = len(mem.get("skills", []))

            log_entry = {
                "ts": _datetime.utcnow().isoformat(),
                "pid": os.getpid(),
                "target_idx": idx,
                "target_smi": target,
                "attempt": attempt,
                "status": stats["status"],
                "best_score": stats["best_score"],
                "n_routes": stats["n_routes"],
                "reward": round(reward, 4),
                "gt_similarity": round(gt_sim, 4),
                "injected_skill_ids": credit_ids,
                "n_injected": len(credit_ids),
                "n_credited": n_credited,
                "added": added,
                "n_merged": n_merged,         # extractions folded into existing skills
                "dedup_s": round(dedup_s, 3), # time in the semantic-dedup LLM calls
                "pruned": removed,            # physically removed by storage cleanup
                "promoted": transitions["promoted"],
                "retired": transitions["retired"],
                "reactivated": transitions["reactivated"],
                "eliminated": transitions["eliminated"],
                "n_skills_before": n_before,
                "n_skills_after": n_after,
                "lock_wait_s": round(lock_wait_s, 3),
                "llm_call_s": round(llm_call_s, 3),
            }
            if llm_error:
                log_entry["error"] = llm_error
            try:
                with _APPEND_LOCK, open(log_path, 'a') as lf:
                    lf.write(json.dumps(log_entry) + "\n")
            except Exception:
                pass

            print(
                f"[skill-memory] pid={os.getpid()} idx={idx} a{attempt} {stats['status']} "
                f"reward={reward:.2f} gt={gt_sim:.2f} credited={n_credited} +{len(added)} "
                f"-{len(removed)} skills={n_before}->{n_after} "
                f"promo={len(transitions['promoted'])} retire={len(transitions['retired'])} "
                f"react={len(transitions['reactivated'])} elim={len(transitions['eliminated'])} "
                f"lock_wait={lock_wait_s:.1f}s llm={llm_call_s:.1f}s"
            )
            # Lock released on context-manager exit

    @staticmethod
    def _estimate_message_tokens(message):
        """GUARANTEED upper bound on input tokens for a chat message list.

        We want to grant the model as much completion as possible (completion =
        context - est_input - margin), so est_input must never be LESS than the true token
        count — an under-estimate over-requests completion and triggers a 400. The old
        chars//3 heuristic under-estimated SMILES-dense prompts and caused exactly that.

        For ASCII content (all our prompts are SMILES + English), BPE emits at most one token
        per character, so the raw character count is a provable upper bound. We use it
        directly, plus a small per-message overhead for the chat-template special/role tokens
        that aren't part of the message content, plus a flat per-image budget. This is mildly
        generous for English-heavy prompts but still leaves ~115-120k completion on a 131072
        context, which no response approaches, while making a context-overflow 400 impossible
        for ASCII inputs."""
        chars = 0
        images = 0
        for m in message:
            content = m.get("content", "")
            if isinstance(content, str):
                chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            chars += len(part.get("text", ""))
                        else:
                            images += 1
            for tc in m.get("tool_calls", []) or []:
                chars += len(str(tc.get("function", {}).get("arguments", "")))
        return chars + 8 * len(message) + 16 + images * 1200

    def _chat_create(self, message, model, completion_cap, context_length, temperature,
                     enable_thinking, tools=None, tool_choice=None):
        """Single chat.completions.create call wrapped in the connection-failure retry/
        dead-server guard. Returns the completion, or None if all attempts failed.

        max_tokens is computed PER CALL from the current message size so the growing agentic
        tool conversation can never push input+completion past the context window (the cause
        of the canary-v5/v6 400 'token count exceeds maximum context length' cascade)."""
        est_input = self._estimate_message_tokens(message)
        # Margin must cover residual estimator error AND the fact that a config may set
        # completion_cap == context_length (older 27B configs did). With an over-estimating
        # est_input + this margin, real_input + max_tokens stays strictly under the limit.
        margin = max(2048, context_length // 50)  # >=2k, ~2% of context
        safe_completion = context_length - est_input - margin
        max_tokens = max(256, min(completion_cap, safe_completion))
        # Sent explicitly (2026-09-26): previously omitted, so SGLang silently
        # filled in Qwen's generation_config defaults (top_p 0.95, top_k 20) and a
        # server/version change could alter sampling unseen. Defaults = those values.
        llm_cfg = self.llm_config.get("llm", {})
        top_p = float(llm_cfg.get("top_p", 0.95))
        top_k = int(llm_cfg.get("top_k", 20))
        # Qwen3.5 model card, non-thinking mode: presence_penalty=1.5 "to reduce
        # endless repetitions". 2026-09-27 smoke test: 23/71 answers (32%) were
        # repetition loops that hit max_tokens with no route. Default 0 = off.
        presence_penalty = float(llm_cfg.get("presence_penalty", 0.0))
        for retry in range(3):
            try:
                kwargs = dict(
                    model=model,
                    messages=message,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking},
                                "top_k": top_k},
                )
                if tools is not None:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = tool_choice or "auto"
                completion = self.llm_client.chat.completions.create(**kwargs)
                self._consecutive_conn_failures = 0  # reset on success
                return completion
            except Exception as e:
                print(f"LLM query attempt {retry+1} failed: {type(e).__name__} {e}")
                if "ConnectionError" in type(e).__name__ or "Connection" in str(e):
                    self._consecutive_conn_failures += 1
                    if self._consecutive_conn_failures >= self._max_consecutive_conn_failures:
                        raise RuntimeError(
                            f"Server appears dead: {self._consecutive_conn_failures} consecutive "
                            f"connection failures. Exiting to avoid wasting walltime."
                        ) from e
        return None

    def query_LLM(self, question, model=None, temperature=0.0, image_smiles=None,
                  image_style: str = "01", tools=None, tool_ctx=None, max_tool_iters=8):
        llm_cfg = self.llm_config.get("llm", {})
        model = model or self.llm_model_name
        # PROMPT ABLATION (COMP9991 study). Applied here because query_LLM is the
        # single choke point every prompt passes through -- init, modification,
        # retries and all run modes. `<PLAN>` alone lives in three files with
        # separate copies of the instruction text, so patching individual
        # templates would half-apply the ablation and silently invalidate the arm.
        # No-op unless a `prompt_ablation:` config block is present.
        _abl = self.llm_config.get("prompt_ablation")
        if _abl:
            import prompt_ablation as _pa
            question = _pa.apply(question, _abl, where="query_LLM")
        # Completion ceiling: a route + reasoning is a few thousand tokens; never request the
        # whole context for output (that guaranteed a 400 once input grew past the headroom).
        # The actual per-call max_tokens is derived from the live message size in _chat_create.
        completion_cap = llm_cfg.get("max_tokens", 8192)
        context_length = llm_cfg.get("context_length", 32768)
        multimodal = llm_cfg.get("multimodal", False)

        # Build user message content — plain text, text + images, or pre-built content items
        if isinstance(question, list):
            # Pre-built content items (e.g., Run C with inline images)
            user_content = question
        elif multimodal and image_smiles:
            from modification_prompt import build_image_content_items
            image_items = build_image_content_items(image_smiles, image_style=image_style)
            user_content = [{"type": "text", "text": question}] + image_items
        else:
            user_content = question

        message = [
            {"role": "system", "content": "You are a retrosynthesis agent who can make multi-step retrosynthesis plans based on your molecule knowledge."},
            {"role": "user", "content": user_content},
        ]

        enable_thinking = llm_cfg.get("enable_thinking", False)

        response = None
        _gen_stats = {}
        _t_call = _time.time()
        tool_trace = []       # one entry per tool call (name/args/result) — agentic path only
        assistant_turns = []  # full thread: model reasoning/text at each loop iteration

        if tools is None:
            # --- Unchanged single-shot path: every existing run (A/B/C/D/X). ---
            completion = self._chat_create(message, model, completion_cap, context_length,
                                           temperature, enable_thinking)
            if completion is not None:
                response = completion.choices[0].message.content
                message.append({"role": "assistant", "content": response})
                _usage = getattr(completion, "usage", None)
                _gen_stats = {
                    "completion_tokens": getattr(_usage, "completion_tokens", None),
                    "prompt_tokens": getattr(_usage, "prompt_tokens", None),
                    # "length" = cut off at llm.max_tokens (runaway generation)
                    "finish_reason": getattr(completion.choices[0], "finish_reason", None),
                }
                if llm_cfg.get("strip_think"):   # off unless set (thinking planner, e.g. Intern-S2)
                    response, _gen_stats["reasoning"] = split_think(response, _gen_stats["finish_reason"])
                    message[-1]["content"] = response
        else:
            # --- Agentic tool-calling loop (Run G): the model may interleave validator
            # tool calls with its reasoning, then emit a final <ROUTE>-bearing answer.
            # tool_ctx is the optimizer instance (provides .inventory/.template_dict/.oracle). ---
            from agentic_tools import execute_tool, build_submit_route, validate_route_list
            # Config can cap tool iterations (tools.max_tool_iters); else use the param default.
            max_tool_iters = self.llm_config.get("tools", {}).get("max_tool_iters", max_tool_iters)
            # "As many iterations as context allows": keep validating until the model submits, the
            # conversation approaches the context window, or the hard backstop. The context guard
            # (not a small fixed cap) is the real limiter — leave room for the final completion.
            context_budget = context_length - completion_cap - 2000
            for i in range(max_tool_iters + 1):
                force_answer = (i == max_tool_iters) or (
                    i > 0 and self._estimate_message_tokens(message) > context_budget)
                if force_answer:
                    # NOTE: a named-function tool_choice ({function: submit_route}) 400s with
                    # SGLang's qwen3_coder parser. "none" is accepted and forces a TEXT answer,
                    # so the finalize message asks for a free-text <ROUTE> block. (Voluntary
                    # submit_route on the auto turns above still produces clean structured routes.)
                    message.append({"role": "user", "content": _AGENTIC_FINALIZE_MSG})
                    tool_choice = "none"
                else:
                    tool_choice = "auto"
                completion = self._chat_create(
                    message, model, completion_cap, context_length, temperature, enable_thinking,
                    tools=tools, tool_choice=tool_choice,
                )
                if completion is None:
                    break
                msg = completion.choices[0].message
                assistant_msg = {"role": "assistant", "content": msg.content or ""}
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    assistant_msg["tool_calls"] = [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in tool_calls
                    ]
                message.append(assistant_msg)

                # Record the model's reasoning/text for this turn so the full thread is
                # inspectable (the 'why' behind each tool call), not just the final answer.
                assistant_turns.append({
                    "iter": i,
                    "reasoning_content": getattr(msg, "reasoning_content", None),
                    "content": msg.content or "",
                    "tool_call_names": [tc.function.name for tc in tool_calls] if tool_calls else [],
                })

                if not tool_calls:
                    # Model emitted a plain answer (possibly a free-text <ROUTE> fallback).
                    response = msg.content
                    break

                # Execute each requested tool and feed results back as role:"tool" messages.
                # submit_route is the TERMINAL action: its structured args become the route
                # (no <ROUTE> regex needed) and the loop ends.
                submitted_route_text = None
                for tc in tool_calls:
                    if tc.function.name == "submit_route":
                        rt, result_str, route_obj = build_submit_route(tc.function.arguments)
                        if rt is not None:
                            # MANDATORY pre-acceptance validation: a submitted route is only
                            # accepted if every step is structurally valid. Catches molecule-set
                            # mismatches / invalid reactions that the GA's loose endpoint check
                            # would otherwise wave through. On failure the model is told exactly
                            # which steps are broken and must fix them and submit again.
                            if tool_ctx is not None and route_obj is not None:
                                val = validate_route_list(route_obj, tool_ctx)
                                if val.get("all_steps_valid"):
                                    submitted_route_text = rt
                                    result_str = json.dumps(
                                        {"submitted": True, "validated": True, "n_steps": val.get("n_steps")})
                                else:
                                    bad = [s for s in val.get("steps", []) if not s.get("ok")]
                                    result_str = json.dumps(
                                        {"submitted": False, "rejected": True,
                                         "reason": "route has invalid step(s); fix them and call submit_route again",
                                         "invalid_steps": bad, "error": val.get("error")})
                            else:
                                submitted_route_text = rt
                    else:
                        result_str = execute_tool(tc.function.name, tc.function.arguments, tool_ctx)
                    message.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})
                    tool_trace.append({
                        "iter": i, "name": tc.function.name,
                        "arguments": tc.function.arguments, "result": result_str,
                    })
                    # Arm moltool, image half (2026-11-19, user: "allow the model to also pull the image of
                    # the molecule via tools"). A tool result is a STRING in the chat protocol, so a picture
                    # cannot be returned from one. The tool therefore records what it wants drawn and the
                    # image is appended here as its own user message with image_url content items, which is
                    # the only shape the server accepts pictures in. Requires the server WITHOUT --text-only
                    # (the vision encoder) and llm.multimodal true, both set by the arm's job/config.
                    _pend = getattr(tool_ctx, "_image_tool_pending", None) if tool_ctx is not None else None
                    if _pend:
                        try:
                            from modification_prompt import build_labeled_image_content_items
                            items = build_labeled_image_content_items(list(_pend))
                            if items:
                                message.append({"role": "user", "content": items})
                        except Exception as _e:
                            print(f"[image_tool] could not attach image: {_e}")
                        finally:
                            tool_ctx._image_tool_pending = None
                if submitted_route_text is not None:
                    response = submitted_route_text
                    break

        # Save LLM query/response to JSONL for later analysis
        if response is not None and hasattr(self, 'args') and self.args and hasattr(self.args, 'output_dir'):
            from datetime import datetime
            log_path = os.path.join(self.args.output_dir, f'llm_responses_{os.getpid()}.jsonl')
            log_entry = {
                'ts': datetime.utcnow().isoformat(),
                'run_mode': self.llm_config.get('run_mode', 'A'),
                'target_idx': getattr(self, 'tb_target_idx', -1),
                # Threads share one pid/file under main.py --workers; keep the
                # worker so per-worker timing (gap/timeout analysis) still works.
                'worker': threading.current_thread().name,
                'latency_s': round(_time.time() - _t_call, 1),
                **_gen_stats,
                'question': question if isinstance(question, str) else [
                    item.get('text', '[image]') if item.get('type') == 'text' else '[image]'
                    for item in question
                ],
                'response': response,
            }
            if tools is not None:
                log_entry['n_tool_calls'] = len(tool_trace)
                log_entry['tool_trace'] = tool_trace
                log_entry['assistant_turns'] = assistant_turns
            with _APPEND_LOCK, open(log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

            # Dedicated human-readable tool log (Run G) — the full thread interleaving the
            # model's reasoning, its tool calls, and the tool results, so you can read it
            # without jq. One block per query_LLM call.
            if tools is not None:
                self._write_tool_log(log_entry['ts'], log_entry['target_idx'],
                                     assistant_turns, tool_trace, response)

        print("=>")
        return message, response

    def _write_tool_log(self, ts, target_idx, assistant_turns, tool_trace, response):
        """Append a human-readable block for one agentic generation to tool_calls_<pid>.log.
        Interleaves, per loop iteration: the model's reasoning, the tool calls it made, and
        each tool's result — in full (tool results are already bounded by the per-tool cap in
        execute_tool, so this matches exactly what the model saw). The complete machine record
        is also in llm_responses_<pid>.jsonl. Best-effort — never raises into the loop."""
        try:
            log_path = os.path.join(self.args.output_dir, f'tool_calls_{os.getpid()}.log')
            # Group tool-call results by iteration for interleaving with the thread.
            calls_by_iter = {}
            for c in tool_trace:
                calls_by_iter.setdefault(c['iter'], []).append(c)

            lines = ["=" * 88,
                     f"ts={ts}  target_idx={target_idx}  n_tool_calls={len(tool_trace)}"]
            for turn in assistant_turns:
                i = turn['iter']
                lines.append(f"\n--- iter {i} ---")
                if turn.get('reasoning_content'):
                    lines.append("  [reasoning] " + str(turn['reasoning_content']))
                if turn.get('content'):
                    lines.append("  [assistant] " + str(turn['content']))
                for c in calls_by_iter.get(i, []):
                    lines.append(f"  >> CALL {c['name']}({c['arguments']})")
                    lines.append(f"     -> {c['result']}")
            lines.append("\n  [FINAL RESPONSE]")
            lines.append("  " + str(response))
            lines.append("")
            with _APPEND_LOCK, open(log_path, 'a') as f:
                f.write("\n".join(lines) + "\n")
        except Exception as e:
            print(f"[tool-log] failed to write tool log: {type(e).__name__}: {e}")
    
    def query_deepseek(self, question):
        # Retained for interface compatibility; routes through the configured LLM backend.
        _message, answer = self.query_LLM(question, temperature=0.8)
        return _message, answer

    def fix_reaction_error(self, routes, stepwise_results):
        # Config `prefer_model_reactants` (default off = upstream). When a template
        # yields several outcomes, upstream keeps the top-ranked one even if the
        # model's own reactants are among the valid outcomes -- the next step then
        # refers to the model's intermediate, which is no longer in the set ("chain
        # drift"). 38% of 200 known-correct 2-3 step routes broke this way
        # (2026-09-28). With the flag on, the model's outcome is kept when the
        # template genuinely produces it; validation is otherwise unchanged.
        _cfg = getattr(self, 'llm_config', {})
        prefer_model = bool(_cfg.get('prefer_model_reactants', False) or _cfg.get('prefer_reproducing_template', False))
        updated_routes = []
        for i in range(len(routes)):
            step_id, is_valid, data = stepwise_results[i]
            if data['reaction_existance'] == True and data['reaction_valid'] == True:
                reaction_smiles = data['reaction']

                #reactants = [Chem.MolFromSmiles(smi) for smi in data['reactants']]
                #products = [Chem.MolFromSmiles(smi) for smi in data['product']]
            
                target_rd = rdchiralReactants(data['product'][0])
                reaction_outputs = run_retro(target_rd, reaction_smiles)
                if len(reaction_outputs) > 1:
                    reaction_outputs = self.rank_reactants(reaction_outputs)
                forced = getattr(self, '_switch_forced', None)   # AOT switch choice: the outcome the model picked
                forced_hit = bool(forced and data['product'] and data['product'][0] == forced['product'])
                if forced_hit and len(reaction_outputs) > 1:
                    want = {sanitize_smiles(smi) for smi in forced['reactants']}
                    for out in reaction_outputs:
                        if {sanitize_smiles(smi) for smi in out} == want:
                            reaction_outputs = [out] + [o for o in reaction_outputs if o is not out]
                            break
                if i == 0 and getattr(self, '_switch_choice_on', False) and not forced_hit:
                    model_set = {sanitize_smiles(smi) for smi in (data.get('reactants') or [])}
                    if not any({sanitize_smiles(x) for x in out} == model_set for out in reaction_outputs):
                        # the template the model's own reaction matched gives other reactants: upstream
                        # writes them in silently; switch choice offers them (near misses) instead
                        self._switch_pending_outcome(routes[i], data, reaction_outputs, reaction_smiles)
                        data['reaction_existance'] = False      # -> check_route_extra rejects step 1
                        continue
                if prefer_model and not forced_hit and len(reaction_outputs) > 1:
                    model_set = {sanitize_smiles(smi) for smi in (data.get('reactants') or [])}
                    for out in reaction_outputs:
                        if model_set and {sanitize_smiles(smi) for smi in out} == model_set:
                            if out is not reaction_outputs[0]:
                                self._fx_bump('fixB_changed')   # upstream would have kept outputs[0]
                            reaction_outputs = [out] + [o for o in reaction_outputs if o is not out]
                            break
                reactants_generated = [reactant for reactant in reaction_outputs[0]]
                reactants_generated = [sanitize_smiles(smi) for smi in reactants_generated]
                reactants_smiles = set(reactants_generated)

                products_smiles = {smi for smi in data['product']}
                original_molecule_set = [Chem.MolFromSmiles(smi) for smi in data['molecule_set']]
                original_set = {Chem.MolToSmiles(mol) for mol in original_molecule_set if mol is not None}
                #print(reactants_smiles)
                updated_mol_set = (original_set | reactants_smiles) - products_smiles
                data['Updated molecule set'] = list(updated_mol_set)
                routes[i]['Reactants'] = str(list(reactants_smiles))
                routes[i]['Updated molecule set'] = str(list(updated_mol_set))

        return routes
    def rank_reactants(self, reactants_list):
        """
        Rank reactants based on the number of products generated
        """
        non_empty_reactant_list = [item for item in reactants_list if item != []]
        scores = [self.oracle.reward(self.inventory, reactant, self.visited_molecules, self.dead_molecules) for reactant in non_empty_reactant_list]
        sorted_list = [x for _, x in sorted(zip(scores, non_empty_reactant_list), key=lambda pair: pair[0], reverse=True)]
        return sorted_list
def _all_matching_template_keys(reaction_smiles, preprocessed_dict, cap=200):
    """All template keys is_reaction_in_dict would accept, in dict order (it
    returns only the first). Same matching rule (utils.is_one_to_one_match)."""
    try:
        products, reactants = reaction_smiles.split(">>")
        reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
        product_mols = [Chem.MolFromSmiles(p) for p in products.split(".")]
    except Exception:
        return []
    if None in reactant_mols or None in product_mols:
        return []
    keys = []
    for key, (smarts_reactant_mols, smarts_product_mols) in preprocessed_dict.items():
        try:
            if len(smarts_reactant_mols) != len(reactant_mols) or len(smarts_product_mols) != len(product_mols):
                continue
            if is_one_to_one_match(smarts_reactant_mols, reactant_mols) and \
                    is_one_to_one_match(smarts_product_mols, product_mols):
                keys.append(key)
                if len(keys) >= cap:
                    break
        except Exception:
            continue
    return keys


def check_and_update_routes(routes, target_list):
    import ast

    routes[0]['Molecule set'] = str(target_list)
    for i in range(1, len(routes)):
        current_updated_set = ast.literal_eval(routes[i]['Molecule set'])
        previous_molecule_set = ast.literal_eval(routes[i - 1]['Updated molecule set'])


        if set(current_updated_set) != set(previous_molecule_set):
            print(f"Mismatch found at step {i}:")
            print(f"  Previous Molecule set: {previous_molecule_set}")
            print(f"  Current Updated molecule set: {current_updated_set}")

            routes[i]['Molecule set'] = str(previous_molecule_set)
            print(f"  Updated step {i} to match previous Molecule set.")

    print("\nAll steps checked and updated where necessary.")
    return routes



def map_reaction(routes, stepwise_results):
    # Use found reactions to substitute the reaction proposed by the LLM
    for i in range(len(routes)):
        step_id, is_valid, data = stepwise_results[i]
        #if data['reaction_existance'] == True:
        reaction_smiles = data['reaction']

        routes[i]['Reaction'] = str([reaction_smiles])

    return routes

def extract_molecules_from_output(output):
    try:

        parsed_output = ast.literal_eval(output)

        if isinstance(parsed_output, list):
            return parsed_output
        elif isinstance(parsed_output, str):
            return [parsed_output]
        else:
            return []
    except (ValueError, SyntaxError):

        return []