from __future__ import print_function

import random
from typing import List

import joblib
import numpy as np
from joblib import delayed
from rdkit import Chem, rdBase
from rdkit.Chem.rdchem import Mol
from rdkit import DataStructs
import ast
from syntheseus import Molecule
import rdkit.Chem.AllChem as AllChem
rdBase.DisableLog('rdApp.error')
from utils import *
from optimizer import BaseOptimizer, check_and_update_routes, map_reaction, extract_molecules_from_output
import os
import openai
import re
import copy
from route import *
from modification_prompt import modification_hints, modification_hints_routes, construct_modification_prompt, construct_init_prompt_C, construct_init_prompt_D, smiles_to_fp_string
from utils import *
import time
import math

MINIMUM = 1e-10

# Run G (agentic): instructions appended to the prompt telling the model the validation
# tools exist and that it should use them to check its proposals at reason time before
# committing the final <ROUTE>. The tool schemas themselves are passed via tools=.
_TOOL_INSTRUCTIONS = """

You have access to tools. Use them WHILE you reason:
- suggest_disconnections: given a product, look up real USPTO reactions used to make SIMILAR molecules — use this when unsure how to break a bond, instead of guessing.
- check_reaction_in_uspto: confirm a backward reaction (Product>>Reactant1.Reactant2) has a USPTO precedent. Every disconnection in the final route must pass this.
- validate_smiles: confirm a SMILES is a real, parseable molecule.
- run_retro: apply a disconnection to a product and see the reactants it produces.
- check_purchasable: check whether molecules are commercially purchasable building blocks (the route is complete only when every leaf is purchasable).
- estimate_complexity: compare how synthetically complex candidate intermediates are (lower is simpler).
- submit_route: SUBMIT YOUR FINAL ROUTE. This is how you finish.

Workflow: when unsure how to disconnect, call suggest_disconnections; confirm a chosen disconnection with check_reaction_in_uspto; check intermediates with check_purchasable. Be efficient — validate only the key/uncertain steps, do not validate every molecule, and do not explore indefinitely.

When you have a route you are reasonably confident in (a PARTIAL route that is not yet fully purchasable is acceptable — it gets refined later), call submit_route with the structured steps. submit_route is the ONLY correct way to finish — do not write the route as plain text. Aim to submit well before exhausting your tool budget.
"""

# Appended only when a skill store is loaded (see _agentic_setup).
_SKILL_TOOL_INSTRUCTION = """- search_skills: when you are unsure how to disconnect an intermediate, retrieve proven high-level disconnection STRATEGIES for its functional groups (distinct strategies, each with worked SMILES examples). Pull these on demand, then adapt one and confirm it with check_reaction_in_uspto.
"""






def extract_route_block(answer):
    """The text between the route tags, stripped, or None if there is no route tag.

    Upstream tried <ROUTE>...<ROUTE> FIRST. When an answer holds two route blocks,
    that pattern spans from the first opening tag to the second, swallowing the
    first block's </ROUTE>, so literal_eval fails (591 of 775 outer parse failures
    in AOT v3). Trying the closing tag first recovers 380 of 8,165 answers with no
    regressions; <ROUTE>...<ROUTE> stays as the fallback for answers that close with
    an opening tag. The strip() avoids IndentationError on a leading newline + indent
    (2026-10-14, docs/experiment_history.md)."""
    m = re.search(r"<ROUTE>(.*?)</ROUTE>", answer, re.DOTALL) or \
        re.search(r"<ROUTE>(.*?)<ROUTE>", answer, re.DOTALL)
    return m.group(1).strip() if m else None


def clean_last_step(route):
    """Drop a no-op final step (upstream's "Route cleaned!" rule): its updated set
    equals the set before it, or it has no reactants.

    Upstream compared against route[-2], which raised IndexError on every ONE-STEP
    route, so all single-step answers were discarded as errors (1,226 of 9,431
    baseline answers, 13%) -- including the natural answer when only one leaf is
    left to make. For one step, "the set before it" is the step's own Molecule set.
    Routes with >= 2 steps behave exactly as upstream.
    """
    comp1 = ast.literal_eval(route[-1]['Updated molecule set'])
    if len(route) >= 2:
        comp2 = ast.literal_eval(route[-2]['Updated molecule set'])
    else:
        comp2 = ast.literal_eval(route[-1]['Molecule set'])
    last_step_reactants = route[-1]['Reactants']
    if set(comp1) == set(comp2) or last_step_reactants == "" or last_step_reactants == "[]" or last_step_reactants == "None" or last_step_reactants == "[None]":
        route = route[:-1]
        print('Route cleaned!')
    if not route:
        raise ValueError('route is empty after cleaning')
    return route


def build_init_prompt_parts(target_repr, examples):
    """The upstream initialization prompt pieces, verbatim (string literals keep
    their original indentation; guarded by test_init_prompt_matches_upstream).
    Shared by planning_Optimizer.initialization and the AOT* search mode."""
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

    task_description = '''
                My target molecule is: {}

                To assist you for the format, an example retrosynthesis route is provided.\n {}

                Please propose a retrosynthesis route for my target molecule. The provided reference routes may be helpful. You can also design a synthetic route based on your own knowledge.
                '''.format(target_repr, examples)

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
    return initial_prompt, task_description, requirements

def make_mating_pool(combined_list: List, population_scores, visited_cache, inventory, offspring_size: int):

    # scores -> probs 
    population_scores = [node_value(combined_list[i], population_scores[i], visited_cache, inventory) for i in range(len(population_scores))]
    weights = np.exp(-np.array(population_scores))  # Exponentially invert scores
    probabilities = weights / weights.sum()  # Normalize to get probabilities
    # Resample until >1 distinct parent, but bounded: with a 1-route population
    # (possible once max_init_retries lets initialization return None) or all
    # but one weight underflowing to 0, the old `while True` spun forever.
    for _ in range(100):
        sampled_index = np.random.choice(len(combined_list), p=probabilities, size=offspring_size, replace=True)
        if len(set(sampled_index)) > 1:
            break
    mating_pool = [combined_list[i] for i in sampled_index]
    return mating_pool

def node_value(combined_list, population_score, visited_cache, inventory, C=0.3):
    molecule_list = combined_list[1].validated_route[-1]['Updated molecule set']
    unpurchasable_list = check_availability(molecule_list, inventory)

    if len(unpurchasable_list) == 0:
        return 1
    total_visits = sum(visited_cache.values())
    node_visits = 1
    for molecule in unpurchasable_list:
        if molecule in visited_cache:
            print(f"visited times: {visited_cache[molecule]}")
            node_visits += visited_cache[molecule]
            if visited_cache[molecule] > 25:
                population_score = -10
    if total_visits == 0:
        final_score = - population_score
    else:
        final_score = - population_score - C * math.sqrt(math.log(total_visits) / node_visits)
        print(f"population_score: {- population_score}")
        print(f"visiting_score: {- C * math.sqrt(math.log(total_visits) / node_visits)}")
    return final_score

class planning_Optimizer(BaseOptimizer):

    def __init__(self, args=None, config=None, shared_data=None):
        super().__init__(args, config=config, shared_data=shared_data)
        self.model_name = "planning"
        self.run_mode = self.llm_config.get("run_mode", "A")

    def _molecule_repr(self, smiles_list):
        """Return (text_repr_list, image_smiles) for the current run_mode.

        run_mode  text shown in prompt              image_smiles
        A         SMILES as-is                      None
        B         SMILES as-is                      smiles_list
        C         SMILES as-is                      smiles_list  (+ labeled images)
        D         image-only LLM2-style; SMILES end-of-prompt write-only ref
                                                    smiles_list
        E         "molecule shown in attached image N"  smiles_list
        F         2048-bit ECFP4 binary string       None
        """
        mode = self.run_mode.upper()
        if mode in ("B", "C", "D", "E"):
            return smiles_list, smiles_list
        elif mode == "F":
            reprs = [smiles_to_fp_string(s) or s for s in smiles_list]
            return reprs, None
        else:  # A, G (both text-only SMILES)
            return smiles_list, None

    def _agentic_setup(self, prompt, phase):
        """Append tool instructions + return tool schemas/context, gated PER PHASE
        (phase = 'init' or 'modification'). Agentic tool-calling is now config-driven via the
        `tools:` block (tools.enabled), NOT tied to run_mode: Run G enables it, and Run S
        (run_mode A + skill memory) opts in the same way. Deep agentic validation on every
        evolution offspring starves the GA of breadth, so tools can be enabled only where they
        help via tools.inject_init / tools.inject_modification (both default True). When tools
        are disabled (no block, or disabled for this phase), returns the prompt unchanged with
        tools=None so query_LLM takes its plain single-shot path."""
        tcfg = self.llm_config.get("tools", {})
        if not tcfg.get("enabled", False):
            return prompt, None, None
        flag = "inject_init" if phase == "init" else "inject_modification"
        if not tcfg.get(flag, True):
            return prompt, None, None
        from agentic_tools import TOOL_SCHEMAS
        # search_skills is only meaningful when a skill store is loaded. Expose it
        # (and describe it) only then, so non-skill runs (e.g. Run G) don't see a
        # tool that always errors. When on, the model can PULL distinct strategies
        # on demand -- the complement to the call-40 PUSH injection.
        skills_on = getattr(self, '_skill_memory_enabled', False) and bool(getattr(self, '_skill_memory', None))
        schemas = TOOL_SCHEMAS if skills_on else [t for t in TOOL_SCHEMAS
                                                  if t.get("function", {}).get("name") != "search_skills"]
        instr = _TOOL_INSTRUCTIONS + (_SKILL_TOOL_INSTRUCTION if skills_on else "")
        if isinstance(prompt, list):
            prompt = prompt + [{"type": "text", "text": instr}]
        else:
            prompt = prompt + instr
        return prompt, schemas, self

    def initialization(self, target_smi, rag_tuples):
        exploration_signal = True
        population = []
        route_list = [t[1] for t in rag_tuples]
        sims_list = [t[0] for t in rag_tuples]


        sum_scores = sum(sims_list)
        population_probs = [p / sum_scores for p in sims_list]
        max_init_retries = getattr(self, '_current_config', {}).get('max_init_retries', None)
        init_attempts = 0
        while True:
            # Honour the per-attempt wall-clock cap here too: it was only checked
            # in the GA loop, so a molecule whose initialization outlasted the
            # 12h walltime never got a .done and was re-attempted every chunk.
            deadline = getattr(self, '_attempt_deadline', None)
            if deadline is not None and time.time() > deadline:
                print('Attempt time limit reached during initialization, skipping.')
                return None
            if max_init_retries is not None:
                if init_attempts >= max_init_retries:
                    print(f'Max init retries ({max_init_retries}) reached, skipping molecule.')
                    return None
                init_attempts += 1
            try:
                sampled_index = np.random.choice(len(route_list), p=population_probs, size=3, replace=False)
                sampled_routes = [route_list[i] for i in sampled_index]
                examples = ''
                for i in sampled_routes:
                    examples = examples + '<ROUTE>\n'+ str(process_reaction_routes(i)) + '\n</ROUTE>\n'
                examples = examples + '\n'
                target_repr, image_smiles = self._molecule_repr([target_smi])
                initial_prompt, task_description, requirements = build_init_prompt_parts(target_repr[0], examples)
                image_style = getattr(self, '_current_config', {}).get('image_style', '01')
                # Strategy guide injection (Run A guided + Run D variants). For Run D
                # the builder consumes strategy_context directly so the guide is placed
                # before the SMILES-notation reference block. For other modes the guide
                # is appended after the prompt is built.
                inject_init = self.llm_config.get('strategy_guide_inject_init', True)
                init_strategy_context = self._strategy_context if (inject_init and getattr(self, '_strategy_context', '')) else ""
                # Run S: bandit-sampled skill memory supersedes the flat guide. The
                # rendered block already carries the === GUIDE === delimiters, so it is
                # appended directly (skip the guide_section wrapper below to avoid
                # double delimiters). init_strategy_context is forced empty so the
                # flat-guide path stays inert when skill memory is enabled.
                skill_block = ""
                if getattr(self, '_skill_memory_enabled', False) and self._skill_cfg.get('inject_init', True):
                    skill_block = self._select_skills_for_injection(target=target_smi)
                    init_strategy_context = ""
                _multimodal = self.llm_config.get('llm', {}).get('multimodal', False)
                if self.run_mode.upper() == "D" and image_smiles and _multimodal:
                    question = construct_init_prompt_D(
                        target_repr[0], sampled_routes,
                        image_style=image_style,
                        strategy_context=init_strategy_context,
                    )
                elif self.run_mode.upper() == "C" and image_smiles:
                    question = construct_init_prompt_C(target_repr[0], examples, image_smiles, image_style=image_style)
                else:
                    question = initial_prompt + requirements + task_description
                # For non-D modes the guide is appended after prompt construction.
                # For Run D the builder already inserted it (skip to avoid duplication).
                if self.run_mode.upper() != "D" and init_strategy_context:
                    guide_section = (
                        "\n=== RETROSYNTHESIS STRATEGY GUIDE ===\n"
                        + init_strategy_context +
                        "\n=== END STRATEGY GUIDE ===\n"
                    )
                    if isinstance(question, str):
                        question = question + guide_section
                    elif isinstance(question, list):
                        question.append({"type": "text", "text": guide_section})
                # Run S: append the pre-delimited bandit-sampled skill block.
                if skill_block:
                    if isinstance(question, str):
                        question = question + skill_block
                    elif isinstance(question, list):
                        question.append({"type": "text", "text": skill_block})
                question, _tools, _tool_ctx = self._agentic_setup(question, "init")
                message, answer = self.query_LLM(question, temperature=0.7, image_smiles=image_smiles, image_style=image_style, tools=_tools, tool_ctx=_tool_ctx)
                #message, answer = self.query_deepseek(question)
                # Converting the extracted string to a Python list of dictionaries
                route_content = extract_route_block(answer)
                if route_content is None:
                    raise ValueError("no <ROUTE> block in answer")

                route = ast.literal_eval(route_content)
                route = clean_last_step(route)  # upstream crashed on 1-step routes
                for step in route:
                    temp = ast.literal_eval(step['Molecule set'])
                    temp = ast.literal_eval(step['Reaction'])[0]
                    products, reactants = temp.split(">>")
                    temp = extract_molecules_from_output(step['Product'])[0]
                    temp = ast.literal_eval(step['Reactants'])[0]
                    temp = ast.literal_eval(step['Updated molecule set'])

                route_class_item = Route(target_smi)
                checked_route, final_evaluation = self.sanitize([target_smi], route, exploration_signal)

                if final_evaluation[0][2]['reaction_existance'] == False and final_evaluation[0][2]['product_inside'] == True:
                    self.update_dead_molecules(final_evaluation[0][2]['product'][0])
                    print('reaction not exist')
                    continue
                if final_evaluation[0][1] == False:
                    print('step invalid')
                    continue
                score = self.rewards(final_evaluation)
                route_class_item.add_route(checked_route, final_evaluation)
                route_class_item.update_reward(score)
                route_class_item.update_evaluation(final_evaluation)
                break
            except RuntimeError:
                raise  # let dead-server errors propagate
            except Exception as e:
                print(f"Error in generating the initial population: {e}")
                continue


        return route_class_item

    def modification(self, combined_list, population_routes, all_fps, route_list, inventory):
        exploration_signal = False
        parent_a = random.choice(combined_list)
        sampled_route = parent_a[1]
        count = 0
        final_route_item = None
        for _ in range(5):
            try:
                count = count + 1
                if count >= 5:
                    count = 0
                    parent_a = random.choice(combined_list)
                    sampled_route = parent_a[1]
                route = sampled_route.validated_route
                new_route_item = copy.deepcopy(sampled_route)
                evaluation = new_route_item.evaluation
                molecule_list = route[-1]['Updated molecule set']
                unpurchasable_list = check_availability(molecule_list, inventory)

                #self.update_visited_molecules(unpurchasable_list)
                
                # Run D (image-only) consumes raw route tuples to render example
                # product images inline. Guarded by multimodal=true so the stale
                # text-only config_D.yaml + run_D_*.pbs scripts (Apr 8, no results)
                # still fall back to the standard text path.
                _mm = self.llm_config.get('llm', {}).get('multimodal', False)
                if self.run_mode.upper() == "D" and _mm:
                    retrieved_routes = modification_hints_routes(unpurchasable_list, all_fps, route_list)
                else:
                    retrieved_routes = modification_hints(unpurchasable_list, all_fps, route_list)

                mol_reprs, image_smiles = self._molecule_repr(unpurchasable_list)
                # When Run D's multimodal guard is off, suppress images so
                # construct_modification_prompt takes the text fallback.
                if self.run_mode.upper() == "D" and not _mm:
                    image_smiles = None
                image_style = getattr(self, '_current_config', {}).get('image_style', '01')
                # Strategy guide injection in modification prompts is gated by
                # strategy_guide_inject_modification (default False). v1 found per-step
                # injection net-negative with a static prescriptive guide (7/17 vs 14/17
                # baseline, see analysis 2026-05-15). v4 retests with a self-updating guide.
                inject_mod = self.llm_config.get('strategy_guide_inject_modification', False)
                mod_strategy_context = self._strategy_context if (inject_mod and getattr(self, '_strategy_context', '')) else ""
                # Descriptive-feedback arm (config `descriptive_feedback`, default off):
                # pass the full per-step evaluation so the prompt can localise the
                # failing step and name the validated prefix.
                _desc_fb = bool(self.llm_config.get('descriptive_feedback', False))
                _full_eval = getattr(new_route_item, 'full_evaluation', None) if _desc_fb else None
                new_q = construct_modification_prompt(unpurchasable_list, retrieved_routes, evaluation, inventory, self.oracle.reaction_cache, molecule_repr_list=mol_reprs, run_mode=self.run_mode, image_smiles=image_smiles, image_style=image_style, strategy_context=mod_strategy_context, full_evaluation=_full_eval, descriptive_feedback=_desc_fb, fb_ctx=(self if _desc_fb else None))
                # Run S: also inject the bandit-sampled skill block into GA
                # evolution prompts when skill_memory.inject_modification is on.
                # Reuses the per-molecule cached block (set at init; reset per
                # molecule in optimize()), so no extra selection/LLM-rank call and
                # the same skills condition both init and modification.
                _mod_on = getattr(self, '_skill_memory_enabled', False) and (
                        self._skill_cfg.get('inject_modification', False)
                        or (self._skill_cfg.get('inject_on_stall', False)
                            and getattr(self, '_stall_inject_active', False)))
                if _mod_on and self._skill_cfg.get('inject_per_connection', False):
                    # PER-CONNECTION mode: retrieve a compact skill for each current
                    # unpurchasable intermediate (mirrors how RAG's modification_hints
                    # retrieves per intermediate) instead of one static whole-target
                    # block. Aligned + small, so it complements RAG rather than
                    # displacing it (the target-level block was 59% of the prompt).
                    skill_block_mod = self._select_skills_per_connection(unpurchasable_list)
                    if skill_block_mod:
                        if isinstance(new_q, str):
                            new_q = new_q + skill_block_mod
                        elif isinstance(new_q, list):
                            new_q.append({"type": "text", "text": skill_block_mod})
                elif _mod_on:
                    # resample=True: re-select per modification (fresh draw under
                    # Thompson; same relevant set under relevance-gating). Reuses
                    # the per-molecule cached relevance -> no extra LLM call.
                    # target is passed for the stall-gated path: with inject_init
                    # off the relevance cache was never built at init, and without
                    # a target the prefilter degrades to an arbitrary store slice.
                    skill_block_mod = self._select_skills_for_injection(
                        target=getattr(self, '_current_target', None), resample=True)
                    if skill_block_mod:
                        if isinstance(new_q, str):
                            new_q = new_q + skill_block_mod
                        elif isinstance(new_q, list):
                            new_q.append({"type": "text", "text": skill_block_mod})
                new_q, _tools, _tool_ctx = self._agentic_setup(new_q, "modification")
                new_m, new_a = self.query_LLM(new_q, temperature=0.7, image_smiles=image_smiles, image_style=image_style, tools=_tools, tool_ctx=_tool_ctx)
                
                print(new_a)
                route_content = extract_route_block(new_a)
                if route_content is None:
                    raise ValueError("no <ROUTE> block in answer")

                new_route = ast.literal_eval(route_content)
                new_route = clean_last_step(new_route)  # upstream crashed on 1-step routes
                for idx, step in enumerate(new_route):
                    temp = ast.literal_eval(step['Molecule set'])
                    reaction = ast.literal_eval(step['Reaction'])[0]
                    products, reactants = reaction.split(">>")
                    product = extract_molecules_from_output(step['Product'])[0]
                    temp = ast.literal_eval(step['Reactants'])[0]
                    temp = ast.literal_eval(step['Updated molecule set'])
                    if idx == 0:
                        if products == product:
                            self.update_cache(product, reaction)

                checked_route, final_evaluation = self.sanitize(molecule_list, new_route, exploration_signal)
                if final_evaluation[0][2]['product_inside'] == True:
                    self.update_visited_molecules(final_evaluation[0][2]['product'])


                if final_evaluation[0][2]['reaction_existance'] == False: 
                    if final_evaluation[0][2]['product_inside'] == True:
                        self.update_dead_molecules(final_evaluation[0][2]['product'][0])
                        print('dead molecule + 1')
                    continue
                new_route_item.update_route(checked_route, final_evaluation)
                if not check_distinct_route(population_routes, new_route_item):
                    print('Exists in the population')
                    continue
                # Score/record the whole chain from the true target, not the
                # extension fragment: fragment-keyed buffer entries scored 0.0
                # for routes that never contained the target (fake passes).
                full_evaluation = self.build_full_evaluation(new_route_item, final_evaluation)
                score = self.rewards(full_evaluation)
                
                new_route_item.update_reward(score)
                new_route_item.update_evaluation(final_evaluation)
                final_route_item = new_route_item
                break
            except RuntimeError:
                raise  # let dead-server errors propagate
            except Exception as e:
                print(f"Error in generating the modification population {e}")

                continue

        return final_route_item

    def _optimize(self, target, route_list, all_fps, config):
        # Expose the RAG precedent index to the agentic suggest_disconnections tool
        # (tool_ctx is this optimizer instance). Stored per-molecule; cheap references.
        self._rag_all_fps = all_fps
        self._rag_route_list = route_list

        file_path = 'stat/'+ str(self.args.dataset_name) + '_' + str(self.args.max_oracle_calls) + '_results.txt'
        #args.dataset_name
        population_class = []
        getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=False)
        similarity_metric = DataStructs.BulkTanimotoSimilarity # BulkDiceSimilarity or BulkTanimotoSimilarity

        target = sanitize_smiles(target)
        fp = getfp(target)
    
        sims = similarity_metric(fp, [fp_ for fp_ in all_fps])

        rag_tuples = list(zip(sims, route_list))
        #rag_tuples = sorted(rag_tuples, key=lambda x: x[0], reverse=True)[:50]
        rag_tuples = sorted(rag_tuples, key=lambda x: x[0], reverse=True)
        # RAG fix (config `rag_top_k`, default off = upstream): see modification_prompt.RAG_TOP_K
        import modification_prompt as _mp
        _mp.RAG_TOP_K = config.get('rag_top_k') or None
        if _mp.RAG_TOP_K:
            rag_tuples = rag_tuples[:_mp.RAG_TOP_K]
        self._current_config = config  # make config accessible to initialization()
        # Walltime resume (see BaseOptimizer._ckpt_save): restore the GA state of
        # a walltime-cut attempt instead of starting it over.
        resume = getattr(self, '_resume_state', None)
        self._resume_state = None
        if resume is not None:
            self._ckpt_restore_common(resume)
        else:
            assert len(self.oracle.route_buffer) == 0, f"route_buffer not empty: {self.oracle.route_buffer}"

        max_evolution_iters = config.get('max_evolution_iters', None)
        if resume is not None and resume['stage'] == 'ga':
            population_routes = resume['population_routes']
            population_scores = resume['population_scores']
            combined_list = resume['combined_list']
            all_routes = resume['all_routes']
            evolution_iter = resume['evolution_iter']
        else:
            starting_population = list(resume['population']) if resume is not None else []
            first_slot = resume['init_slots_done'] if resume is not None else 0
            for slot in range(first_slot, config["population_size"]):
                r = self.initialization(target, rag_tuples)
                if r is not None:
                    starting_population.append(r)
                self._ckpt_save('init', population=starting_population, init_slots_done=slot + 1)

            if not starting_population:
                print(f'No valid initial routes found for {target}, skipping.')
                return

            # initial population
            population_routes = starting_population

            population_scores = [route_class_item.get_reward() for route_class_item in population_routes]

            combined_list = list(zip(population_scores, population_routes))
            all_routes = copy.deepcopy(population_routes)
            evolution_iter = 0
        # Stall-gated skill injection (supervisor note 2026-07-17): free
        # exploration runs unguided FIRST; skills turn on only for the final
        # stretch of molecules still unsolved. Two triggers (whichever fires
        # first), then injection stays on for the rest of the molecule:
        #   (a) inject_after_calls (default 40): a HARD, deterministic trigger --
        #       once the buffer reaches N oracle calls without a solve, inject.
        #       This is the primary gate. It exists because the plateau signal
        #       (b) almost never fired: on pistachio_hard the best score CREEPS
        #       (-6.4 -> -6.3 -> ...) toward an unreachable 0.0, so it rarely
        #       goes stall_window calls flat even while failing (only 1/9 mols
        #       stalled in the 2026-07-20 run). Calls 1..N stay unguided, so
        #       deep free exploration is preserved and any molecule that solves
        #       before N (via the pass-found break) never sees a skill.
        #   (b) stall_window plateau: kept as an EARLIER opportunistic trigger
        #       for the rare genuine plateau; harmless when it doesn't fire.
        _skill_cfg = getattr(self, '_skill_cfg', {}) if getattr(self, '_skill_memory_enabled', False) else {}
        _stall_gate_on = bool(_skill_cfg.get('inject_on_stall', False))
        _stall_window = int(_skill_cfg.get('stall_window', 15))
        _inject_after = int(_skill_cfg.get('inject_after_calls', 40))
        self._stall_inject_active = False
        _stall_best, _stall_mark = None, 0
        if resume is not None and resume['stage'] == 'ga':
            self._stall_inject_active = resume.get('stall_inject_active', False)
            _stall_best, _stall_mark = resume.get('stall_best'), resume.get('stall_mark', 0)
        while True:
            if max_evolution_iters is not None and evolution_iter >= max_evolution_iters:
                print(f'Max evolution iterations ({max_evolution_iters}) reached, moving on.')
                break
            # Per-attempt wall-clock cap (Run S v2 retry-to-solve): stop this GA
            # attempt cleanly once its deadline passes so it counts as a completed
            # (failed) attempt and the retry loop can move on / persist progress.
            deadline = getattr(self, '_attempt_deadline', None)
            if deadline is not None and time.time() > deadline:
                print(f'Attempt time limit reached ({(time.time()-getattr(self,"_attempt_start",deadline))/3600:.2f}h), moving on.')
                break
            evolution_iter += 1

            if _stall_gate_on and not self._stall_inject_active and self.oracle.route_buffer:
                _best_now = max(float(v[0]) for v in self.oracle.route_buffer.values())
                _n_now = len(self.oracle.route_buffer)
                if _stall_best is None or _best_now > _stall_best:
                    _stall_best, _stall_mark = _best_now, _n_now
                _hit_budget = _inject_after > 0 and _n_now >= _inject_after
                _hit_plateau = (_n_now - _stall_mark) >= _stall_window
                if _hit_budget or _hit_plateau:
                    self._stall_inject_active = True
                    _why = (f'reached {_n_now} oracle calls (>= {_inject_after})' if _hit_budget
                            else f'best score {_stall_best:.3f} unimproved for '
                                 f'{_n_now - _stall_mark} calls (>= {_stall_window})')
                    print(f'[stall-gate] {_why} at evolution_iter={evolution_iter}, '
                          f'best={_stall_best:.3f} (unsolved) -> skill injection ON')
                    # Give the skills a real budget to work with: injecting at 40
                    # of a 50-call budget leaves only ~10 guided calls. Extend so
                    # the guided phase gets a full run. NB this makes total budget
                    # exceed the baseline's -- any comparison needs a baseline at
                    # the same extended budget to stay honest (see CLAUDE.md).
                    _bonus = int(_skill_cfg.get('stall_budget_bonus', 0))
                    if _bonus > 0:
                        self.oracle.max_oracle_calls += _bonus
                        print(f'[stall-gate] oracle budget extended by {_bonus} -> '
                              f'{self.oracle.max_oracle_calls} total')

            if len(self.oracle) > 5:
                self.sort_buffer()
                # Pass detection must use the SAME authoritative per-route score
                # the save loop below uses (route.get_reward(), i.e. evaluate()),
                # not the proxy population_scores (oracle.reward on the final
                # molecule set only). The proxy can report 0 when the authoritative
                # score does not, causing a "found a route" abort that saves nothing
                # and ends the molecule as a false EARLY_TERM. (Ranking still uses
                # the proxy at line ~501 — only the stop/save decision is changed.)
                if any(route.get_reward() == 0 for route in population_routes):
                    self.log_intermediate(finish=True) 
                    print('find a route, abort ...... ')
                    for route in population_routes:
                        reward = route.get_reward()
                        if reward == 0:
                            route.save_result(target)
                            
                    break


            # new_population
            mating_pool = make_mating_pool(combined_list, population_scores, self.visited_molecules, self.inventory, config["population_size"])

            #modification
            offspring_routes = []
            for _ in range(config["offspring_size"]):
                result = self.modification(mating_pool, all_routes, all_fps, route_list, self.inventory)
                if result is not None:
                    if check_distinct_route(offspring_routes, result):
                        offspring_routes.append(result)

            # add new_population
            population_routes += offspring_routes
            all_routes = all_routes + offspring_routes
            # stats
            old_scores = population_scores


            all_scores = [self.oracle.reward(self.inventory, route_class_item.validated_route[-1]['Updated molecule set'],self.visited_molecules, self.dead_molecules) for route_class_item in all_routes]
            combined_list = list(zip(all_scores, all_routes))
            combined_list = sorted(combined_list, key=lambda x: x[0], reverse=True)[:config["population_size"]]
            population_routes = [t[1] for t in combined_list]
            population_scores = [t[0] for t in combined_list]

            # TensorBoard: log best population score at each iteration
            if getattr(self, 'tb_writer', None) is not None and population_scores:
                best = min(population_scores)
                global_step = self.tb_target_idx * 1000 + len(self.oracle)
                self.tb_writer.add_scalar('iteration/best_score', best, global_step)

            self._ckpt_save('ga', population_routes=population_routes,
                            population_scores=population_scores,
                            combined_list=combined_list, all_routes=all_routes,
                            evolution_iter=evolution_iter,
                            stall_inject_active=self._stall_inject_active,
                            stall_best=_stall_best, stall_mark=_stall_mark)



            ### stopping

            if len(self.oracle) > 5:
                print(population_scores)
                # Pass detection must use the SAME authoritative per-route score
                # the save loop below uses (route.get_reward(), i.e. evaluate()),
                # not the proxy population_scores (oracle.reward on the final
                # molecule set only). The proxy can report 0 when the authoritative
                # score does not, causing a "found a route" abort that saves nothing
                # and ends the molecule as a false EARLY_TERM. (Ranking still uses
                # the proxy at line ~501 — only the stop/save decision is changed.)
                if any(route.get_reward() == 0 for route in population_routes):
                    self.log_intermediate(finish=True)  
                    print('Find route, abort ...... ')
                    for route in population_routes:
                        reward = route.get_reward()
                        if reward == 0:
                            route.save_result(target)
                            
                    break
  
            if self.finish:   
                print('finished, abort ...... ')              
                break