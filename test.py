import datasets

refs = [ # First set of references
        ['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.'],
        ['The dog had bit the man.', 'No one was surprised.', 'The man had bitten the dog.'],
]
sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']

# bleu = datasets.load_metric("sacrebleu")
# hypothesis = [itm.strip().split() for itm in sys]
# reference = [[itm.strip().split()] for itm in refs]
# res = bleu.compute(predictions=hypothesis, references=reference)
# print(res)


predictions = ["hello there", "general kenobi", "big distance"]
references = ["hello there", "general kenobi", "big distance"]

bleurt = datasets.load_metric("bleurt", 'bleurt-large-512')
results = bleurt.compute(predictions=predictions, references=references)

print(results)