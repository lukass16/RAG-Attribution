"""
Simple test script to demonstrate accessing questions and contexts from the synthetic data.
"""

from data_loader import SyntheticDataLoader

loader = SyntheticDataLoader()

print("\n\n" + "="*80)
print("EXAMPLE: Complementary Context")
print("="*80)

df = loader.load_csv('20_complementary.csv')
pairs = loader.get_question_context_pairs(df)

question, contexts, answer = pairs[0]

print(f"\nQUESTION:")
print(f"  {question}")

# Check if contexts2 is a list of strings or needs further parsing
if isinstance(contexts, list) and len(contexts) > 0:
    if isinstance(contexts[0], str):
        # Try to parse if it's a JSON string
        import json
        try:
            contexts = json.loads(contexts[0])
        except:
            pass

print(f"\nCONTEXT DOCUMENTS ({len(contexts)} total):")
for i, context_doc in enumerate(contexts, 1):
    print(f"\n  Document {i}:")
    print(f"    {context_doc}")

print(f"\nANSWER:")
print(f"  {answer}")



# Show how to iterate through all data
print("\n\n" + "="*80)
print("ITERATION EXAMPLE: Processing All Questions")
print("="*80)

print("\nProcessing first 3 questions from 20_complementary.csv:")
for idx, (q, ctx_list, ans) in enumerate(pairs[:3], 1):
    print(f"\n{idx}. Q: {q[:80]}...")
    print(f"   Contexts: {len(ctx_list)} documents")
    print(f"   A: {ans[:80]}...")


