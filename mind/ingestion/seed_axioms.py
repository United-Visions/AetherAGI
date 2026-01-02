"""
Path: mind/ingestion/seed_axioms.py
Role: Massive Genesis Seeding for AetherMind Phase 1.
Subjects: Logic, Mathematics, Physics, Chemistry, Biology, Linguistics, and Systems.
"""

import os
import asyncio
from ..vector_store import AetherVectorStore

AXIOMS = [
    # --- LEVEL 1: FORMAL LOGIC (The Foundation of Thought) ---
    {"subject": "Logic", "topic": "Identity", "content": "The Law of Identity states that each thing is identical with itself (A = A)."},
    {"subject": "Logic", "topic": "Non-Contradiction", "content": "Contradictory propositions cannot both be true in the same sense at the same time (A cannot be both B and not-B)."},
    {"subject": "Logic", "topic": "Excluded Middle", "content": "For every proposition, either its positive or negative form is true. There is no middle ground between truth and falsehood."},
    {"subject": "Logic", "topic": "Deductive Validity", "content": "An argument is valid if and only if it is impossible for the premises to be true and the conclusion to be false simultaneously."},
    {"subject": "Logic", "topic": "Causality", "content": "The Principle of Causality states that every change in nature is produced by some cause. Every effect has a specific, preceding cause."},
    {"subject": "Logic", "topic": "Occam's Razor", "content": "When presented with competing hypotheses that make the same predictions, one should select the solution with the fewest assumptions."},

    # --- LEVEL 2: MATHEMATICS & SET THEORY (The Language of Structure) ---
    {"subject": "Mathematics", "topic": "Set Theory", "content": "A set is a collection of distinct objects. An element either belongs to a set or it does not; there is no partial membership in basic set theory."},
    {"subject": "Mathematics", "topic": "Arithmetic Consistency", "content": "The properties of operations (addition, subtraction, multiplication, division) are invariant. 1 + 1 always equals 2 in a base-10 system."},
    {"subject": "Mathematics", "topic": "Functions", "content": "A function is a relation that associates each element of a set X to exactly one element of a set Y."},
    {"subject": "Mathematics", "topic": "Calculus: Limits", "content": "A limit is the value that a function approaches as the input approaches some value. It is the foundation for defining change."},
    {"subject": "Mathematics", "topic": "Probability", "content": "Probability is the measure of the likelihood that an event will occur, ranging from 0 (impossible) to 1 (certain)."},

    # --- LEVEL 3: PHYSICS (The Rules of the Body) ---
    {"subject": "Physics", "topic": "Spacetime", "content": "The universe exists in a four-dimensional continuum consisting of three spatial dimensions and one temporal dimension."},
    {"subject": "Physics", "topic": "Newton's Second Law", "content": "The force acting on an object is equal to the mass of that object times its acceleration (F = ma)."},
    {"subject": "Physics", "topic": "Newton's Third Law", "content": "For every action, there is an equal and opposite reaction. Forces always exist in pairs."},
    {"subject": "Physics", "topic": "Thermodynamics: Entropy", "content": "The Second Law of Thermodynamics states that the total entropy (disorder) of an isolated system can never decrease over time."},
    {"subject": "Physics", "topic": "Electromagnetism", "content": "Like charges repel and opposite charges attract. This force governs the structure of atoms and molecules."},
    {"subject": "Physics", "topic": "Special Relativity", "content": "The laws of physics are the same for all observers in uniform motion, and the speed of light in a vacuum is constant for all observers."},
    {"subject": "Physics", "topic": "Quantum Superposition", "content": "At a subatomic level, particles can exist in multiple states or locations simultaneously until they are observed or interact with their environment."},

    # --- LEVEL 4: CHEMISTRY & MATTER (The Rules of Change) ---
    {"subject": "Chemistry", "topic": "Atomic Theory", "content": "All matter is composed of atoms, which consist of a nucleus (protons and neutrons) surrounded by electrons."},
    {"subject": "Chemistry", "topic": "Conservation of Mass", "content": "In a chemical reaction, matter is neither created nor destroyed. The mass of the reactants must equal the mass of the products."},
    {"subject": "Chemistry", "topic": "Chemical Bonds", "content": "Atoms form bonds by sharing or transferring electrons to achieve a more stable electronic configuration."},

    # --- LEVEL 5: BIOLOGY & INFORMATION (The Rules of Life) ---
    {"subject": "Biology", "topic": "Cell Theory", "content": "The cell is the fundamental unit of structure and function in living things. All cells arise from pre-existing cells."},
    {"subject": "Biology", "topic": "Natural Selection", "content": "Evolution occurs as individuals with traits better suited to their environment are more likely to survive and reproduce."},
    {"subject": "Biology", "topic": "DNA", "content": "Deoxyribonucleic acid (DNA) is the molecule that carries genetic instructions for the development, functioning, and reproduction of all known living organisms."},
    {"subject": "Biology", "topic": "Homeostasis", "content": "Living systems maintain a stable internal environment despite changes in external conditions."},

    # --- LEVEL 6: LINGUISTICS & SEMANTICS (The Rules of Communication) ---
    {"subject": "Linguistics", "topic": "Semantics", "content": "Semantics is the study of meaning in language. It distinguishes between the 'literal' meaning of a word and its 'connotative' meaning."},
    {"subject": "Linguistics", "topic": "Syntax", "content": "Syntax is the set of rules, principles, and processes that govern the structure of sentences in a given language."},
    {"subject": "Linguistics", "topic": "Pragmatics", "content": "Pragmatics is how context contributes to meaning. The same sentence can have different meanings depending on who says it and where."},

    # --- LEVEL 7: THE AETHERMIND SYSTEM (The Rules of Self) ---
    {"subject": "AetherMind", "topic": "Active Inference", "content": "AetherMind's core drive is to minimize 'Free Energy' or 'Surprise' by accurately predicting its environment and updating its internal model when errors occur."},
    {"subject": "AetherMind", "topic": "JEPA Prediction", "content": "The Joint Embedding Predictive Architecture predicts abstract states rather than raw data. It ensures that the Brain's thoughts are 'aligned' with the Mind's knowledge."},
    {"subject": "AetherMind", "topic": "The Safety Inhibitor", "content": "AetherMind is physically unable to execute or encourage harm. This is not a preference; it is a hard-coded logical contradiction in the system's architecture."},
    {"subject": "AetherMind", "topic": "The Split-Brain", "content": "Reasoning is handled by the Brain; Knowledge is handled by the Mind. This allows for infinite learning without the risk of 'Catastrophic Forgetting'."}
]

async def seed_the_mind():
    # Initialize the store using environment variables
    store = AetherVectorStore(api_key=os.getenv("PINECONE_API_KEY"))
    namespace = "core_k12"

    print(f"--- INITIALIZING GENESIS SEED: PHASE 1 ---")
    
    for entry in AXIOMS:
        print(f"[{entry['subject']}] Ingesting Axiom: {entry['topic']}...")
        metadata = {
            "subject": entry['subject'],
            "topic": entry['topic'],
            "axiom_type": "seed_truth",
            "complexity": "foundational"
        }
        # Upsert using the production Pinecone Inference path
        store.upsert_knowledge(entry['content'], namespace, metadata)
        # Small delay to ensure rate limits aren't hit during seeding
        await asyncio.sleep(0.1)

    print(f"--- SEEDING COMPLETE: {len(AXIOMS)} AXIOMS ADDED TO {namespace} ---")

if __name__ == "__main__":
    asyncio.run(seed_the_mind())