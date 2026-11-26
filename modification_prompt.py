import base64
import copy
import ast
from io import BytesIO
from itertools import permutations
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
from syntheseus import Molecule
from rdkit import DataStructs
import numpy as np
from multiprocessing import Pool, cpu_count
from rdchiral.main import rdchiralRun
from rdchiral.initialization import rdchiralReactants, rdchiralReaction
from utils import *
import random


def smiles_to_base64_image(smiles: str, size: int = 300, image_style: str = "01") -> str | None:
    """Render a SMILES string to a base64-encoded PNG using the given image style.

    Styles:
      "01" — default PIL MolToImage (original behaviour)
      "06" — Cairo renderer with stereo annotations
      "15" — Cairo renderer with aromatic atoms/bonds highlighted in orange
      "20" — 2×2 grid of 4 ETKDG conformer projections; falls back to "01" on failure
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        rdDepictor.Compute2DCoords(mol)

        if image_style == "06":
            d = rdMolDraw2D.MolDraw2DCairo(size, size)
            d.drawOptions().addStereoAnnotation = True
            d.DrawMolecule(mol)
            d.FinishDrawing()
            return base64.b64encode(d.GetDrawingText()).decode("utf-8")

        elif image_style == "15":
            arom_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
            arom_bonds = [b.GetIdx() for b in mol.GetBonds() if b.GetIsAromatic()]
            if arom_atoms:
                orange = (1.0, 0.6, 0.1)
                atom_cols = {i: orange for i in arom_atoms}
                bond_cols = {i: orange for i in arom_bonds}
                d = rdMolDraw2D.MolDraw2DCairo(size, size)
                d.DrawMolecule(mol,
                               highlightAtoms=arom_atoms,
                               highlightBonds=arom_bonds,
                               highlightAtomColors=atom_cols,
                               highlightBondColors=bond_cols)
            else:
                d = rdMolDraw2D.MolDraw2DCairo(size, size)
                d.DrawMolecule(mol)
            d.FinishDrawing()
            return base64.b64encode(d.GetDrawingText()).decode("utf-8")

        elif image_style == "19":
            try:
                mol_3d = Chem.AddHs(Chem.MolFromSmiles(smiles))
                p = AllChem.ETKDGv3()
                p.randomSeed = 42
                if AllChem.EmbedMolecule(mol_3d, p) == -1:
                    raise ValueError("Embedding failed")
                AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=2000)
                mol_3d_noh = Chem.RemoveHs(mol_3d)
                mol_proj = Chem.RWMol(mol_3d_noh)
                rdDepictor.GenerateDepictionMatching3DStructure(mol_proj, mol_3d_noh, confId=0)
                d = rdMolDraw2D.MolDraw2DCairo(size, size)
                d.drawOptions().addStereoAnnotation = True
                d.DrawMolecule(mol_proj)
                d.FinishDrawing()
                return base64.b64encode(d.GetDrawingText()).decode("utf-8")
            except Exception:
                img = Draw.MolToImage(mol, size=(size, size))
                buf = BytesIO()
                img.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode("utf-8")

        elif image_style == "20":
            try:
                conf_mols = []
                for seed in [0, 7, 42, 99]:
                    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
                    p = AllChem.ETKDGv3()
                    p.randomSeed = seed
                    if AllChem.EmbedMolecule(m, p) == -1:
                        raise ValueError(f"Embedding failed for seed {seed}")
                    AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
                    m = Chem.RemoveHs(m)
                    proj = Chem.RWMol(m)
                    rdDepictor.GenerateDepictionMatching3DStructure(proj, m, confId=0)
                    conf_mols.append(proj)
                grid_img = Draw.MolsToGridImage(conf_mols, molsPerRow=2,
                                                subImgSize=(size, size))
                buf = BytesIO()
                grid_img.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception:
                # Fall back to default PIL rendering if conformer generation fails
                img = Draw.MolToImage(mol, size=(size, size))
                buf = BytesIO()
                img.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode("utf-8")

        else:  # "01" or any unrecognised style
            img = Draw.MolToImage(mol, size=(size, size))
            buf = BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

    except Exception:
        return None


def smiles_to_fp_string(smiles: str) -> str | None:
    """Return 2048-bit ECFP4 binary string for a SMILES (e.g. '10110010...'). Returns None on failure."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        return fp.ToBitString()
    except Exception:
        return None


def build_image_content_items(smiles_list: list, image_style: str = "01") -> list:
    """Return a list of OpenAI image_url content items for each valid SMILES."""
    items = []
    for smi in smiles_list:
        if not smi:
            continue
        b64 = smiles_to_base64_image(smi, image_style=image_style)
        if b64:
            items.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
    return items


def build_labeled_image_content_items(smiles_list: list) -> list:
    """Return image content items with text labels describing each structure.

    Produces: [intro_text, label_1, image_1, label_2, image_2, ...]
    so the model knows what each image represents.
    """
    items = [{"type": "text", "text": (
        "The following images show the 2D molecular structures of the "
        "molecules referenced above. Each image is labeled with its "
        "corresponding SMILES string."
    )}]
    for i, smi in enumerate(smiles_list, 1):
        if not smi:
            continue
        b64 = smiles_to_base64_image(smi)
        if b64:
            items.append({"type": "text", "text": f"Image {i}: 2D structure of {smi}"})
            items.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
    return items


def build_inline_molecule_content(smiles_list, image_style: str = "01"):
    """Return content items with labeled molecule images for inline placement in prompt text."""
    items = []
    for i, smi in enumerate(smiles_list, 1):
        if not smi:
            continue
        b64 = smiles_to_base64_image(smi, image_style=image_style)
        if b64:
            items.append({"type": "text", "text": f"Molecule {i} — {smi}:"})
            items.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
    return items


def change_to_forward_reaction(reaction:str):
    products, reactants = reaction.split(">>")
    return reactants + ">>" + products

def process_reaction_routes(route:list):
    json_list = []
    for idx,i in enumerate(route):
        products, reactants = i.split(">>")
        reaction = i
        reactants_list = reactants.split(".")
        #forward_reaction = change_to_forward_reaction(reaction)
        if idx == 0:
            step = {
                "Molecule set": str([products]),
                "Product": str([products]),
                "Reaction": str([reaction]),
                "Reactants": str(reactants_list),
                "Updated molecule set": str(reactants_list)
            }
            json_list.append(step)
        else:
            #print(json_list)
            original_set = copy.deepcopy(ast.literal_eval(json_list[idx-1]["Updated molecule set"]))
            update_set = copy.deepcopy(ast.literal_eval(json_list[idx-1]["Updated molecule set"]))
            #print(update_set)
            try:
                update_set.remove(products)
            except:
                print(update_set)
                print(products)
            update_set = update_set + reactants_list
            step = {
                "Molecule set": str(original_set),
                "Product": str([reaction]),
                "Reaction": str([reaction]),
                "Reactants": str(reactants_list),
                "Updated molecule set": str(update_set)
            }
            json_list.append(step)
    
    return json_list

def check_availability(smi_list, inventory):
    unavailable_list = []
    for smi in smi_list:
        signal = inventory.is_purchasable(Molecule(smi))
        if not signal:
            unavailable_list.append(smi)
    
    return unavailable_list
    

# 2026-10-07 RAG fix (config `rag_top_k`, set per run by planning_Optimizer._optimize).
# Upstream samples in proportion to similarity over ALL ~364k routes, which makes the
# shown examples ~random (median similarity 0.15 vs 0.46 for the top 3; P(any of the
# 100 nearest) ~0.1%). Upstream's own commented-out line and the AOT* release keep
# only the 50 nearest. None = upstream behaviour.
RAG_TOP_K = None


def retrieve_routes(target_smi, all_fps, route_list, number):
    getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=False)
    similarity_metric = DataStructs.BulkTanimotoSimilarity # BulkDiceSimilarity or BulkTanimotoSimilarity
    fp = getfp(target_smi)
    sim_score = similarity_metric(fp, [fp_ for fp_ in all_fps])

    rag_tuples = list(zip(sim_score, route_list))
    #rag_tuples = sorted(rag_tuples, key=lambda x: x[0], reverse=True)[:50]
    rag_tuples = sorted(rag_tuples, key=lambda x: x[0], reverse=True)
    if RAG_TOP_K:   # config `rag_top_k` (default off = upstream): sample only among the K nearest
        rag_tuples = rag_tuples[:RAG_TOP_K]
    route_list = [t[1] for t in rag_tuples]
    sims_list = [t[0] for t in rag_tuples]


    sum_scores = sum(sims_list)
    population_probs = [p / sum_scores for p in sims_list]
    #population_probs = [1/len(sims_list) for p in sims_list]
    sampled_index = np.random.choice(len(route_list), p=population_probs, size=number, replace=False)
    sampled_routes = [route_list[i] for i in sampled_index]

    return sampled_routes

def modification_hints(unperchasable_molecule, all_fps, route_list):

    fb = '''\n
    '''
    examples = []
    for smi in unperchasable_molecule:
        examples = examples + retrieve_routes(smi, all_fps, route_list, 3)
    random.shuffle(examples)
    for i in examples:
        fb = fb + str(process_reaction_routes(i)) + '\n'

    return fb

def redundant_reaction(molecule_smi_list, reaction_cache):
    text = ''
    for idx, smi in enumerate(molecule_smi_list):
        if smi in reaction_cache:
            smi_description = 'For molecule {}, these reactions have been tried previously:\n'.format(smi)
            for reaction in reaction_cache[smi]:
                smi_description += reaction + '\n' 
            text += smi_description 
            text += 'Please do not use them again.\n'
    return text

def construct_init_prompt_C(target_repr, examples, image_smiles, image_style: str = "01"):
    """Build Run C initialization prompt as content items with inline molecule images."""

    initial_prompt = '''
                You are a professional chemist specializing in synthesis analysis. Your task is to propose a retrosynthesis route for a target molecule provided in SMILES format.

                Definition:
                A retrosynthesis route is a sequence of backward reactions that starts from the target molecules and ends with commercially purchasable building blocks.

                Key concepts:
                - Molecule set: The working set of molecules at any given step. Initially, it contains only the target molecule.
                - Commercially purchasable: Molecules that can be directly bought from suppliers (permitted building blocks).
                - Non-purchasable: Molecules that must be further decomposed via retrosynthesis steps.
                - Reaction source: All reactions must be derived from the USPTO dataset, and stereochemistry (e.g., E/Z isomers, chiral centers) must be preserved.

                Process:
                1. Initialization: Start with the molecule set = [target molecule].
                2. Iteration:
                    - Select one non-purchasable molecule from the molecule set (the product).
                    - Apply a valid backward reaction from the USPTO dataset to decompose it into reactants.
                    - Remove the product molecule from the set.
                    - Add the reactants to the set.
                3. Termination: Continue until all molecules in the set are commercially purchasable.
            '''

    requirements = '''
                You need to analyze the target molecule and make a retrosynthesis plan in the <PLAN></PLAN> before proposing the route. After making the plan, you should explain the plan in the <EXPLANATION></EXPLANATION>. The route should be a list of steps wrapped in <ROUTE></ROUTE>. Each step in the list should be a dictionary.
                At the first step, the molecule set should be the target molecules set given by the user. Here is an example:

                <PLAN>: Analyze the target molecule and plan for each step in the route. </PLAN>
                <EXPLANATION>: Explain the plan. </EXPLANATION>
                <ROUTE>
                [
                    {
                        'Molecule set': "['Target Molecule']",
                        'Rational': 'Step analysis',
                        'Product': "['Product molecule']",
                        'Reaction': "['Reaction template']",
                        'Reactants': "['Reactant1', 'Reactant2']",
                        'Updated molecule set': "['Reactant1', 'Reactant2']"
                    },
                    {
                        'Molecule set': "['Reactant1', 'Reactant2']",
                        'Rational': 'Step analysis',
                        'Product': "['Product molecule']",
                        'Reaction': "['Reaction template']",
                        'Reactants': "['subReactant1', 'subReactant2']",
                        'Updated molecule set': "['Reactant1', 'subReactant1', 'subReactant2']"
                    }
                ]
                </ROUTE>

                \n\n
                Requirements: 1. The 'Molecule set' contains molecules we need to synthesize at this stage. In the first step, it should be the target molecule. In the following steps, it should be the 'Updated molecule set' from the previous step.\n
                2. The 'Rational' part in each step should be your analysis for syhthesis planning in this step. It should be in the string format wrapped with \'\'\n
                3. 'Product' is the molecule we plan to synthesize in this step. It should be from the 'Molecule set'. The molecule should be a molecule from the 'Molecule set' in a list. The molecule smiles should be wrapped with \'\'.\n
                4. 'Reaction' is a backward reaction which can decompose the product molecule into its reactants. The reaction should be in a list. All the molecules in the reaction template should be in SMILES format. For example, ['Product>>Reactant1.Reactant2'].\n
                5. 'Reactants' are the reactants of the reaction. It should be in a list. The molecule smiles should be wrapped with \'\'.\n
                6. The 'Updated molecule set' should be molecules we need to purchase or synthesize after taking this reaction. To get the 'Updated molecule set', you need to remove the product molecule from the 'Molecule set' and then add the reactants in this step into it. In the last step, all the molecules in the 'Updated molecule set' should be purchasable.\n
                7. In the <PLAN>, you should analyze the target molecule and plan for the whole route.\n
                8. In the <EXPLANATION>, you should analyze the plan.\n'''

    molecule_intro = '''
                My target molecule is: {}

                A labeled 2D structural diagram is provided below for reference. It may be useful for verifying functional groups, confirming ring topology, or resolving any ambiguity in the SMILES string:
            '''.format(target_repr)

    inline_images = build_inline_molecule_content(image_smiles, image_style=image_style)

    task_rest = '''
                To assist you for the format, an example retrosynthesis route is provided.\n {}

                Please propose a retrosynthesis route for my target molecule. The provided reference routes may be helpful. You can also design a synthetic route based on your own knowledge.
            '''.format(examples)

    content_items = [
        {"type": "text", "text": initial_prompt + requirements + molecule_intro},
    ] + inline_images + [
        {"type": "text", "text": task_rest},
    ]

    return content_items


def _construct_modification_prompt_C(molecule_smi_list, repr_text, examples, evaluation,
                                     previous_route, feedback, reaction_cache, image_smiles,
                                     image_style: str = "01"):
    """Build Run C modification prompt as content items list with inline molecule images."""

    initial_prompt = '''
        You are a professional chemist specializing in synthesis analysis. Your task is to propose or modify a retrosynthesis route for target molecules provided in SMILES format.

        Definition:
        A retrosynthesis route is a sequence of backward reactions that starts from the target molecules and ends with commercially purchasable building blocks.

        Key concepts:
        - Molecule set: The working set of molecules at any given step. Initially, it contains only the target molecules.
        - Commercially purchasable: Molecules that can be directly bought from suppliers (permitted building blocks).
        - Non-purchasable: Molecules that must be further decomposed via retrosynthesis steps.
        - Reaction source: All reactions must be derived from the USPTO dataset, and stereochemistry (e.g., E/Z isomers, chiral centers) must be preserved.

        Process:
        1. Initialization: Start with the molecule set = {target molecules}.
        2. Iteration:
            - Select one non-purchasable molecule from the molecule set (the product).
            - Apply a valid backward reaction from the USPTO dataset to decompose it into reactants.
            - Remove the product molecule from the set.
            - Add the reactants to the set.
        3. Termination: Continue until all molecules in the set are commercially purchasable.
    '''

    requirements = '''
    You need to analyze the target molecule and make a retrosynthesis plan in the <PLAN></PLAN> before proposing the route.  After making the plan, you should explain the plan in the <EXPLANATION></EXPLANATION>. The route should be a list of steps wrapped in <ROUTE></ROUTE>. Each step in the list should be a dictionary. You need to keep a molecule set in which are the molecules we need to synthesize or purchase. In each step, you need to select a molecule from the 'Molecule set'
    as the prodcut molecule in this step and use a reaction to synthesize it. Usually, the reactants are eariser to synthesize or can be purchased from the market. After proposing the reaction in this step, you need to remove the product molecule from the molecule set and add the reactants in this reaction into the molecule set and then name
    this updated set as the 'Updated molecule set' in this step. In the next step, the starting molecule set should be the 'Updated molecule set' from the previous step. In the last step, all the molecules in the 'Updated molecule set' should be purchasable. Here is an example:
    corresponds to a set of molecules that are commercially available. Here is an example:

    <PLAN>: Analyze the target molecule set and plan for each step in the route. </PLAN>
    <EXPLANATION>: Explaination for the whole route. </EXPLANATION>
    <ROUTE>
    [
        {
            'Molecule set': "['Target molecules']",
            'Rational': 'Step analysis',
            'Product': "['Product molecule']",
            'Reaction': "['Reaction template']",
            'Reactants': "['Reactant1', 'Reactant2']",
            'Updated molecule set': "['Reactant1', 'Reactant2']"
        },
        {
            'Molecule set': "['Reactant1', 'Reactant2']",
            'Rational': 'Step analysis',
            'Product': "['Product molecule']",
            'Reaction': "['Reaction template']",
            'Reactants': "['subReactant1', 'subReactant2']",
            'Updated molecule set': "['Reactant1', 'subReactant1', 'subReactant2']"
        }
    ]
    </ROUTE>
    \n\n
    Requirements: 1. The 'Molecule set' contains all of the molecules we need to synthesize. In the first step, it should be the list of target molecules given by the user. In the following steps, it should be the 'Updated molecule set' from the previous step.\n
    2. The 'Rational' part in each step should be your analysis for syhthesis planning in this step. It should be in the string format wrapped with \'\'\n
    3. 'Product' is the molecule we plan to synthesize in this step. It should be from the 'Molecule set'. The molecule should be a molecule from the 'Molecule set' in a list. The molecule smiles should be wrapped with \'\'.\n
    4. 'Reaction' is a backward reaction which can decompose the product molecule into its reactants. The reaction should be in a list. All the molecules in the reaction template should be in SMILES format. For example, ['Product>>Reactant1.Reactant2'].\n
    5. 'Reactants' are the reactants of the reaction. It should be in a list. The molecule smiles should be wrapped with \'\'.\n
    6. The 'Updated molecule set' should be molecules we need to purchase or synthesize after taking this reaction. To get the 'Updated molecule set', you need to remove the product molecule from the 'Molecule set' and then add the reactants in this step into it. In the last step, all the molecules in the 'Updated molecule set' should be purchasable.\n
    7. In the <PLAN>, you should make a plan to synthesize the target molecules.\n
    8. In the <EXPLANATION>, you should explain the plan.\n'''

    molecule_intro = '''
        My target molecule set is: {}

        Labeled 2D structural diagrams of these molecules are provided below for reference. They may be useful for verifying functional groups, confirming ring topology, or resolving any ambiguity in the SMILES strings:
    '''.format(repr_text)

    inline_images = build_inline_molecule_content(image_smiles, image_style=image_style)

    if evaluation['product_inside'] == False:
        task_rest = '''
        To assist you for the format, an example retrosynthesis route is provided.\n {}

        Please propose a retrosynthesis route for the target molecule set. The provided reference routes may be helpful. You can also design a synthetic route based on your own knowledge.
        '''.format(examples)
    else:
        task_rest = '''
        In the previous attempt, the first step is:
        <ROUTE>
        {}
        </ROUTE>
        The feedback for this step is: {}
        To assist you for the format, an example retrosynthesis route is provided.\n {}
        Please propose a retrosynthesis route for the starting molecule set. The provided reference routes may be helpful. You can also design a synthetic route based on your own knowledge. All the molecules should be in SMILES format. For example, Cl2 should be ClCl in SMILES format. Br2 should be BrBr in SMILES format. H2O should be O in SMILES format. HBr should be [H]Br in SMILES format. NH3 should be N in SMILES format. Hydrogen atoms are implicitly understood unless explicitly needed for clarity.
        '''.format(previous_route, feedback, examples)

    # Assemble content items: text → inline images → rest
    content_items = [
        {"type": "text", "text": initial_prompt + requirements + molecule_intro},
    ] + inline_images + [
        {"type": "text", "text": task_rest},
    ]

    # Add redundant reaction checking
    reduntant_checking = False
    for smi in molecule_smi_list:
        if smi in reaction_cache:
            reduntant_checking = True
            break
    if reduntant_checking:
        content_items.append({"type": "text", "text": redundant_reaction(molecule_smi_list, reaction_cache)})

    return content_items


def construct_modification_prompt(molecule_smi_list, examples, evaluation, inventory, reaction_cache, molecule_repr_list=None, run_mode=None, image_smiles=None, image_style: str = "01", strategy_context: str = "", full_evaluation=None, descriptive_feedback=False, fb_ctx=None):
    initial_prompt = '''
        You are a professional chemist specializing in synthesis analysis. Your task is to propose or modify a retrosynthesis route for target molecules provided in SMILES format.

        Definition:
        A retrosynthesis route is a sequence of backward reactions that starts from the target molecules and ends with commercially purchasable building blocks.

        Key concepts:
        - Molecule set: The working set of molecules at any given step. Initially, it contains only the target molecules.
        - Commercially purchasable: Molecules that can be directly bought from suppliers (permitted building blocks).
        - Non-purchasable: Molecules that must be further decomposed via retrosynthesis steps.
        - Reaction source: All reactions must be derived from the USPTO dataset, and stereochemistry (e.g., E/Z isomers, chiral centers) must be preserved.

        Process:
        1. Initialization: Start with the molecule set = {target molecules}.
        2. Iteration:
            - Select one non-purchasable molecule from the molecule set (the product).
            - Apply a valid backward reaction from the USPTO dataset to decompose it into reactants.
            - Remove the product molecule from the set.
            - Add the reactants to the set.
        3. Termination: Continue until all molecules in the set are commercially purchasable.
    '''
    # molecule_repr_list overrides the text shown for molecules in the prompt:
    # - None / Runs A-D: show SMILES as-is
    # - Run E: ["molecule shown in attached image 1", ...]
    # - Run F: ["<2048-bit fp string>", ...]
    repr_list = molecule_repr_list if molecule_repr_list is not None else molecule_smi_list
    repr_text = str(repr_list)

    previous_route, feedback = get_feedback(molecule_smi_list, evaluation, inventory)
    # Step-localised feedback arm: tell the model WHICH step failed and which
    # prefix already validated, so it repairs rather than rewrites. Off by default.
    if descriptive_feedback:
        import descriptive_feedback as _df
        feedback = _df.enhance(feedback, evaluation, full_evaluation, inventory, ctx=fb_ctx)

    # Run C: return content items with inline molecule images
    if run_mode and run_mode.upper() == "C" and image_smiles:
        return _construct_modification_prompt_C(
            molecule_smi_list, repr_text, examples, evaluation,
            previous_route, feedback, reaction_cache, image_smiles,
            image_style=image_style
        )

    # Run D: image-only LLM2-style prompt (no SMILES in plan/explanation;
    # SMILES at end as write-only notation reference). Requires examples to be
    # the raw retrieved-routes list (not the pre-joined text blob).
    if run_mode and run_mode.upper() == "D" and image_smiles:
        return _construct_modification_prompt_D(
            molecule_smi_list, examples, evaluation, inventory,
            reaction_cache, image_smiles, image_style=image_style,
            strategy_context=strategy_context,
        )

    if evaluation['product_inside'] == False:
        task_description = '''
        My target molecule set is: {}

        To assist you for the format, an example retrosynthesis route is provided.\n {}

        Please propose a retrosynthesis route for the target molecule set. The provided reference routes may be helpful. You can also design a synthetic route based on your own knowledge.
        '''.format(repr_text, examples)
    else:
        task_description = '''
        My target molecule set is: {}

        In the previous attempt, the first step is:
        <ROUTE>
        {}
        </ROUTE>
        The feedback for this step is: {}
        To assist you for the format, an example retrosynthesis route is provided.\n {}
        Please propose a retrosynthesis route for the starting molecule set. The provided reference routes may be helpful. You can also design a synthetic route based on your own knowledge. All the molecules should be in SMILES format. For example, Cl2 should be ClCl in SMILES format. Br2 should be BrBr in SMILES format. H2O should be O in SMILES format. HBr should be [H]Br in SMILES format. NH3 should be N in SMILES format. Hydrogen atoms are implicitly understood unless explicitly needed for clarity.
        '''.format(repr_text, previous_route, feedback, examples)
    

    requirements = '''
    You need to analyze the target molecule and make a retrosynthesis plan in the <PLAN></PLAN> before proposing the route.  After making the plan, you should explain the plan in the <EXPLANATION></EXPLANATION>. The route should be a list of steps wrapped in <ROUTE></ROUTE>. Each step in the list should be a dictionary. You need to keep a molecule set in which are the molecules we need to synthesize or purchase. In each step, you need to select a molecule from the 'Molecule set'
    as the prodcut molecule in this step and use a reaction to synthesize it. Usually, the reactants are eariser to synthesize or can be purchased from the market. After proposing the reaction in this step, you need to remove the product molecule from the molecule set and add the reactants in this reaction into the molecule set and then name
    this updated set as the 'Updated molecule set' in this step. In the next step, the starting molecule set should be the 'Updated molecule set' from the previous step. In the last step, all the molecules in the 'Updated molecule set' should be purchasable. Here is an example:
    corresponds to a set of molecules that are commercially available. Here is an example:
    
    <PLAN>: Analyze the target molecule set and plan for each step in the route. </PLAN>
    <EXPLANATION>: Explaination for the whole route. </EXPLANATION>
    <ROUTE>
    [   
        {
            'Molecule set': "['Target molecules']",
            'Rational': 'Step analysis',
            'Product': "['Product molecule']",
            'Reaction': "['Reaction template']",
            'Reactants': "['Reactant1', 'Reactant2']",
            'Updated molecule set': "['Reactant1', 'Reactant2']"
        },
        {
            'Molecule set': "['Reactant1', 'Reactant2']",
            'Rational': 'Step analysis',
            'Product': "['Product molecule']",
            'Reaction': "['Reaction template']",
            'Reactants': "['subReactant1', 'subReactant2']",
            'Updated molecule set': "['Reactant1', 'subReactant1', 'subReactant2']"
        }
    ]
    </ROUTE>
    \n\n
    Requirements: 1. The 'Molecule set' contains all of the molecules we need to synthesize. In the first step, it should be the list of target molecules given by the user. In the following steps, it should be the 'Updated molecule set' from the previous step.\n
    2. The 'Rational' part in each step should be your analysis for syhthesis planning in this step. It should be in the string format wrapped with \'\'\n
    3. 'Product' is the molecule we plan to synthesize in this step. It should be from the 'Molecule set'. The molecule should be a molecule from the 'Molecule set' in a list. The molecule smiles should be wrapped with \'\'.\n
    4. 'Reaction' is a backward reaction which can decompose the product molecule into its reactants. The reaction should be in a list. All the molecules in the reaction template should be in SMILES format. For example, ['Product>>Reactant1.Reactant2'].\n
    5. 'Reactants' are the reactants of the reaction. It should be in a list. The molecule smiles should be wrapped with \'\'.\n
    6. The 'Updated molecule set' should be molecules we need to purchase or synthesize after taking this reaction. To get the 'Updated molecule set', you need to remove the product molecule from the 'Molecule set' and then add the reactants in this step into it. In the last step, all the molecules in the 'Updated molecule set' should be purchasable.\n
    7. In the <PLAN>, you should make a plan to synthesize the target molecules.\n
    8. In the <EXPLANATION>, you should explain the plan.\n'''

    question = initial_prompt + requirements + task_description
    reduntant_checking = False
    for smi in molecule_smi_list:
        if smi in reaction_cache:
            reduntant_checking = True
            break
    if reduntant_checking:
        question += redundant_reaction(molecule_smi_list, reaction_cache)
    if strategy_context:
        question += (
            "\n=== RETROSYNTHESIS STRATEGY GUIDE ===\n"
            + strategy_context +
            "\n=== END STRATEGY GUIDE ===\n"
        )
    return question


# ---------------------------------------------------------------------------
# Run D — image-only LLM2-style prompt builders.
#
# Ported from LLM-Syn-Planner-TriFusion/triLLM_prompt.py LLM2 builders.
# Run D shows the target molecule(s) as images (no SMILES alongside) and
# enforces vision-first reasoning in <PLAN>/<EXPLANATION>. The SMILES string
# is provided ONLY at the very end of the prompt as a labelled write-only
# notation reference for constructing SMARTS inside <ROUTE>.
# ---------------------------------------------------------------------------

_RUN_D_RETRO_DEFINITION = '''
You are a professional chemist specializing in retrosynthesis analysis.

Definition:
A retrosynthesis route is a sequence of backward reactions that starts from the target molecule and ends with commercially purchasable building blocks.

Key concepts:
- Molecule set: The working set of molecules at any given step. Initially, it contains only the target molecule.
- Commercially purchasable: Molecules that can be directly bought from chemical suppliers.
- Non-purchasable: Molecules that must be further decomposed via retrosynthesis steps.
- Reaction source: All reactions must be derived from the USPTO dataset. Stereochemistry must be preserved.

Process:
1. Initialization: Start with the molecule set = [target molecule].
2. Iteration: Select one non-purchasable molecule, apply a backward USPTO reaction to decompose it, update the molecule set.
3. Termination: Continue until all molecules in the set are commercially purchasable.
'''

_RUN_D_MOD_RETRO_DEFINITION = '''
You are a professional chemist specializing in retrosynthesis analysis. Your task is to propose a retrosynthesis route for a set of target molecules that are not yet commercially purchasable.

Definition:
A retrosynthesis route is a sequence of backward reactions that starts from the target molecules and ends with commercially purchasable building blocks.

Key concepts:
- Molecule set: The working set of molecules at any given step.
- Commercially purchasable: Molecules that can be directly bought from chemical suppliers.
- Non-purchasable: Molecules that must be further decomposed.
- Reaction source: All reactions must be from the USPTO dataset. Stereochemistry must be preserved.

Process:
1. Start with the molecule set = {given target molecules}.
2. Iteration: Select one non-purchasable molecule, apply a backward USPTO reaction, update the molecule set.
3. Termination: Continue until all molecules are commercially purchasable.
'''

_RUN_D_ROUTE_FORMAT_REQUIREMENTS = '''
You need to analyze the target molecule and make a retrosynthesis plan in the <PLAN></PLAN> before proposing the route. After making the plan, you should explain the plan in the <EXPLANATION></EXPLANATION>. The route should be a list of steps wrapped in <ROUTE></ROUTE>. Each step in the list should be a dictionary.
At the first step, the molecule set should be the target molecules set given by the user. Here is an example:

<PLAN>: Analyze the target molecule and plan for each step in the route. </PLAN>
<EXPLANATION>: Explain the plan. </EXPLANATION>
<ROUTE>
[
    {
        'Molecule set': "['TargetSMILES']",
        'Rational': 'Step analysis string',
        'Product': "['ProductSMILES']",
        'Reaction': "['ProductSMILES>>Reactant1SMILES.Reactant2SMILES']",
        'Reactants': "['Reactant1SMILES', 'Reactant2SMILES']",
        'Updated molecule set': "['Reactant1SMILES', 'Reactant2SMILES']"
    },
    {
        'Molecule set': "['Reactant1SMILES', 'Reactant2SMILES']",
        'Rational': 'Step analysis string',
        'Product': "['Reactant1SMILES']",
        'Reaction': "['Reactant1SMILES>>subReactant1.subReactant2']",
        'Reactants': "['subReactant1', 'subReactant2']",
        'Updated molecule set': "['subReactant1', 'subReactant2', 'Reactant2SMILES']"
    }
]
</ROUTE>

Requirements:
1. The 'Molecule set' contains molecules we need to synthesize at this stage. In the first step, it should be the target molecule. In the following steps, it should be the 'Updated molecule set' from the previous step.
2. The 'Rational' part in each step should be your analysis for synthesis planning in this step. It should be in the string format wrapped with ''.
3. 'Product' is the molecule we plan to synthesize in this step. It should be from the 'Molecule set'. The molecule should be in a list. The molecule SMILES should be wrapped with ''.
4. 'Reaction' is a backward reaction which can decompose the product molecule into its reactants. The reaction should be in a list. All the molecules in the reaction template should be in SMILES format. For example, ['Product>>Reactant1.Reactant2'].
5. 'Reactants' are the reactants of the reaction. It should be in a list. The molecule SMILES should be wrapped with ''.
6. The 'Updated molecule set' should be molecules we need to purchase or synthesize after taking this reaction. To get the 'Updated molecule set', you need to remove the product molecule from the 'Molecule set' and then add the reactants in this step into it. In the last step, all the molecules in the 'Updated molecule set' should be purchasable.
7. CRITICAL: The <ROUTE> block must be a valid Python list literal that can be parsed by ast.literal_eval(). Use single quotes for strings inside dicts. No trailing commas. No Python expressions.
'''


def modification_hints_routes(unperchasable_molecule, all_fps, route_list, per_mol=3):
    """Run D variant of modification_hints — returns the raw route tuples
    (not joined into text) so the image-only prompt builder can render each
    example's product structure inline alongside the route text.
    """
    examples = []
    for smi in unperchasable_molecule:
        examples = examples + retrieve_routes(smi, all_fps, route_list, per_mol)
    random.shuffle(examples)
    return examples


def construct_init_prompt_D(target_smi, sampled_routes, image_style: str = "01",
                             strategy_context: str = ""):
    """Run D initialization prompt — image-only, LLM2-style.

    Target molecule shown as image first; SMILES placed at the end as a
    write-only notation reference. RAG examples include both the example
    product image and its route text.
    """
    content = []
    content.append({"type": "text", "text": _RUN_D_RETRO_DEFINITION})
    content.append({"type": "text", "text": (
        "Your task is to analyze the target molecule shown as an image below "
        "and propose a retrosynthesis route based on its visual structure.\n\n"
        "VISION-FIRST REASONING — HARD CONSTRAINTS:\n"
        "  • <PLAN> and <EXPLANATION> MUST describe the molecule using visual features only: "
        "ring systems and their sizes, substituent positions, hetero atom colours, stereo "
        "wedge/dash bonds, bond orders you can see, and connectivity in the rendered image.\n"
        "  • Do NOT parse, quote, or refer to the SMILES character string in your reasoning. "
        "Do NOT mention 'SMILES' in <PLAN> or <EXPLANATION>.\n"
        "  • The SMILES string is shown only at the very end of this prompt, "
        "labelled as a notation reference. Treat it like a checksum: a write-only resource "
        "you may consult ONLY when constructing the reaction SMARTS inside <ROUTE>. "
        "It is not a source of truth about structure."
    )})
    content.append({"type": "text", "text": _RUN_D_ROUTE_FORMAT_REQUIREMENTS})

    # Target image — present first, prominently, with no SMILES alongside.
    content.append({"type": "text", "text": "=== TARGET MOLECULE (image only) ==="})
    content.append({"type": "text", "text": (
        "Analyze the following molecular structure image. This is the molecule "
        "you need to decompose into purchasable building blocks. Describe what "
        "you SEE, not what a string says:"
    )})
    b64_target = smiles_to_base64_image(target_smi, image_style=image_style)
    if b64_target:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_target}"},
        })
    else:
        content.append({"type": "text", "text": "[Image generation failed for target molecule]"})

    # RAG examples — image + route text per example
    if sampled_routes:
        content.append({"type": "text", "text": (
            f"=== REFERENCE EXAMPLES ===\n"
            f"The following {len(sampled_routes)} examples show retrosynthesis routes "
            f"for visually similar molecules. Use them as format and strategy references:"
        )})
        for i, route in enumerate(sampled_routes, 1):
            try:
                product_smi = route[0].split(">>")[0]
                route_text = str(process_reaction_routes(route))
                content.append({"type": "text",
                                 "text": f"--- Example {i}: A similar molecule ---"})
                b64_ex = smiles_to_base64_image(product_smi, image_style=image_style)
                if b64_ex:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_ex}"},
                    })
                content.append({"type": "text",
                                 "text": f"Known retrosynthesis route for Example {i}:\n{route_text}"})
            except Exception as e:
                content.append({"type": "text",
                                 "text": f"[Example {i}: Could not render — {e}]"})

    # Self-updating strategy guide injection point (init prompts only).
    if strategy_context:
        content.append({"type": "text", "text": (
            "\n=== RETROSYNTHESIS STRATEGY GUIDE ===\n"
            + strategy_context +
            "\n=== END STRATEGY GUIDE ===\n"
        )})

    content.append({"type": "text", "text": (
        "=== YOUR TASK ===\n"
        "Based on your VISUAL analysis of the target molecule and the reference routes above, "
        "propose a retrosynthesis route. Output your <PLAN>, <EXPLANATION>, and <ROUTE>.\n\n"
        "SELF-CHECK before submitting: scan your <PLAN> and <EXPLANATION>. If either "
        "contains the literal word 'SMILES', or quotes any character substring from the "
        "reference SMILES below verbatim, rewrite that passage to describe what you SEE "
        "in the image (rings, substituents, stereo) instead. Visual reasoning, not string "
        "parsing."
    )})

    content.append({"type": "text", "text": (
        f"=== SMILES NOTATION REFERENCE (write-only — use ONLY inside <ROUTE> for SMARTS) ===\n"
        f"{target_smi}\n"
        f"=== END SMILES REFERENCE ==="
    )})

    return content


def _construct_modification_prompt_D(unpurchasable_list, raw_retrieved_routes,
                                      evaluation, inventory, reaction_cache,
                                      image_smiles, image_style: str = "01",
                                      strategy_context: str = ""):
    """Run D modification prompt — image-only, LLM2-style.

    unpurchasable_list: list of SMILES still needing decomposition.
    raw_retrieved_routes: list of raw route tuples (NOT the joined text blob).
    image_smiles: same SMILES used to render images (usually == unpurchasable_list).
    """
    content = []
    content.append({"type": "text", "text": _RUN_D_MOD_RETRO_DEFINITION})
    content.append({"type": "text", "text": (
        "VISION-FIRST REASONING — HARD CONSTRAINTS:\n"
        "  • <PLAN> and <EXPLANATION> MUST describe each molecule using visual features only: "
        "ring systems and sizes, substituent positions, hetero atom colours, stereo bonds, "
        "and connectivity in the rendered images.\n"
        "  • Do NOT parse, quote, or refer to the SMILES character strings in your reasoning. "
        "Do NOT mention 'SMILES' in <PLAN> or <EXPLANATION>.\n"
        "  • The SMILES strings appear only at the end of this prompt, labelled as a "
        "notation reference. Treat them like checksums: write-only resources you consult "
        "ONLY when constructing reaction SMARTS inside <ROUTE>. They are not sources of "
        "truth about structure."
    )})
    content.append({"type": "text", "text": _RUN_D_ROUTE_FORMAT_REQUIREMENTS})

    # Unpurchasable molecule images (no SMILES alongside)
    content.append({"type": "text", "text": (
        "=== TARGET MOLECULES (images only) ===\n"
        "The following molecules are not yet commercially purchasable and must be decomposed. "
        "Analyze their visual structures — describe what you SEE, not what a string says:"
    )})
    for i, smi in enumerate(image_smiles, 1):
        content.append({"type": "text", "text": f"Molecule {i}:"})
        b64 = smiles_to_base64_image(smi, image_style=image_style)
        if b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        else:
            content.append({"type": "text", "text": f"[Image failed for molecule {i}]"})

    # Previous-attempt feedback (chemically specific; text-only)
    if isinstance(evaluation, dict) and evaluation.get('product_inside', False):
        try:
            previous_route, feedback = get_feedback(unpurchasable_list, evaluation, inventory)
            content.append({"type": "text", "text": (
                "=== PREVIOUS ATTEMPT FEEDBACK ===\n"
                f"First step attempted: {previous_route}\n"
                f"Feedback: {feedback}\n"
                "Please correct the issues identified above."
            )})
        except Exception:
            pass

    # RAG examples — image + route text per example
    if raw_retrieved_routes:
        content.append({"type": "text", "text": (
            f"=== REFERENCE ROUTES ===\n"
            f"The following {len(raw_retrieved_routes)} reference routes are for visually "
            f"similar molecules. Use them for format and strategy guidance:"
        )})
        for i, route in enumerate(raw_retrieved_routes, 1):
            try:
                product_smi = route[0].split(">>")[0]
                route_text = str(process_reaction_routes(route))
                content.append({"type": "text", "text": f"--- Reference {i} ---"})
                b64_ex = smiles_to_base64_image(product_smi, image_style=image_style)
                if b64_ex:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_ex}"},
                    })
                content.append({"type": "text", "text": f"Route: {route_text}"})
            except Exception as e:
                content.append({"type": "text",
                                 "text": f"[Reference {i}: Could not render — {e}]"})

    # Redundant-reaction warning
    needs_check = any(smi in reaction_cache for smi in unpurchasable_list)
    if needs_check:
        redundant_text = redundant_reaction(unpurchasable_list, reaction_cache)
        if redundant_text.strip():
            content.append({"type": "text", "text": (
                "=== PREVIOUSLY TRIED REACTIONS ===\n" + redundant_text
            )})

    # Self-updating guide injection (defaults to off for modification, but
    # supported via strategy_guide_inject_modification config flag).
    if strategy_context:
        content.append({"type": "text", "text": (
            "\n=== RETROSYNTHESIS STRATEGY GUIDE ===\n"
            + strategy_context +
            "\n=== END STRATEGY GUIDE ===\n"
        )})

    content.append({"type": "text", "text": (
        "=== YOUR TASK ===\n"
        "Based on the molecule IMAGES and reference routes, propose a retrosynthesis route "
        "for the target molecules. Output your <PLAN>, <EXPLANATION>, and <ROUTE>.\n\n"
        "SELF-CHECK before submitting: scan your <PLAN> and <EXPLANATION>. If either "
        "contains the literal word 'SMILES', or quotes any character substring from the "
        "reference SMILES below verbatim, rewrite that passage to describe what you SEE "
        "in the images (rings, substituents, stereo) instead. Visual reasoning, not "
        "string parsing."
    )})

    smi_block_lines = [
        "=== SMILES NOTATION REFERENCE (write-only — use ONLY inside <ROUTE> for SMARTS) ==="
    ]
    for i, smi in enumerate(unpurchasable_list, 1):
        smi_block_lines.append(f"Molecule {i}: {smi}")
    smi_block_lines.append("=== END SMILES REFERENCE ===")
    content.append({"type": "text", "text": "\n".join(smi_block_lines)})

    return content


def get_feedback(molecule_set, evaluation, inventory):
    original_set = copy.deepcopy(molecule_set)
    update_set = copy.deepcopy(original_set)
    try:
        update_set.remove(evaluation['product'][0])
    except:
        update_set = []
    update_set = update_set + evaluation['reactants']
    reaction = evaluation['reaction']
    #forward_reaction = change_to_forward_reaction(evaluation['reaction'])     
    step = [{
            "Molecule set": str(molecule_set),
            "Product": str(evaluation['product']),
            "Reaction": str([reaction]),
            "Reactants": str(evaluation['reactants']),
            "Updated molecule set": str(update_set)
        }]

    feedback = '''\n'''

    # reaction not exist
    if evaluation['reaction_existance'] == False:
        feedback += reaction_unavailable_feedback(evaluation)
                            
    # molecule in valid
    if len(evaluation['invalid_updated_mol_id']) != 0:
        feedback += molecule_invalid_feedback(evaluation)

    if evaluation['reaction_existance'] == True and evaluation['reaction_valid'] == False:
        feedback += reaction_can_not_happen(evaluation)
    
    # runreaction error
    if evaluation['reaction_existance'] == True and evaluation['reaction_valid'] == True and evaluation['updated_set_valid'] == False:
        feedback += updated_set_mismatch_feedback(evaluation)

        
    # molecule unavailable
    if evaluation['check_availability'] == True and len(evaluation['unavailable_mol_id']) != 0 and len(evaluation['invalid_updated_mol_id']) == 0:
        feedback += molecule_unavailable_feedback(evaluation, inventory)

    
    if feedback == '''\n''':
        feedback = 'The provided route is valid. Please try to propose a new synthetic route for this target molecule.'
                


    return str(step), feedback