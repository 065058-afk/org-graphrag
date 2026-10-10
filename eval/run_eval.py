import json
import csv
import os
import time


def score_run(query_fn, output_csv, eval_path="eval/questions.json", limit=None):
    questions = json.load(open(eval_path))
    if limit:
        questions = questions[:limit]

    fieldnames = ["question", "answer", "expected", "source_dept", "seconds"]

    done = set()
    if os.path.exists(output_csv):
        with open(output_csv, newline="") as f:
            done = {row["question"] for row in csv.DictReader(f)}
    write_header = not os.path.exists(output_csv)

    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for i, q in enumerate(questions, 1):
            if q["question"] in done:
                continue
            start = time.time()
            try:
                answer = str(query_fn(q["question"]))
            except Exception as e:
                answer = f"ERROR: {e}"
            elapsed = round(time.time() - start, 1)
            writer.writerow({
                "question": q["question"],
                "answer": answer,
                "expected": q["reference"],
                "source_dept": q["source_dept"],
                "seconds": elapsed,
            })
            f.flush()
            print(f"[{i}/{len(questions)}] {elapsed}s - {q['question'][:60]}", flush=True)

    print(f"Saved results to {output_csv}")
