import evaluate

def evaluate_captions(predictions, references):
    results = {}

    # BLEU
    bleu = evaluate.load("bleu")
    bleu_result = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
    results["BLEU"] = bleu_result["bleu"]

    # ROUGE-L
    rouge = evaluate.load("rouge")
    rouge_result = rouge.compute(predictions=predictions, references=references)
    results["ROUGE-L"] = rouge_result["rougeL"]

    # METEOR
    meteor = evaluate.load("meteor")
    meteor_result = meteor.compute(predictions=predictions, references=references)
    results["METEOR"] = meteor_result["meteor"]

    return results