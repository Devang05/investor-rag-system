import re
import sys
import json
import argparse
from pathlib import Path
from update_db import get_vector_store
from chat import make_client, ask, add_to_history

EVAL_FILE = Path(__file__).resolve().parent / "eval_set.json"

# phrases that show the model admitted the evidence is not enough
INSUFFICIENT_PHRASES = [
    "not enough", "insufficient", "does not specify", "does not define",
    "does not state", "does not mention", "does not provide", "does not contain",
    "not specified", "not provided", "no information", "cannot answer", "unable to answer"
]

# checks for one test case -> list of failure reasons (empty list = pass)
def check_case(case, results, answer):
    failures = []
    answer_lower = (answer or "").lower()

    # retrieval: every expected file must appear in the retrieved chunks
    retrieved_files = {Path(r.metadata["source"]).name for r in results}
    for source in case.get("expected_sources", []):
        if source not in retrieved_files:
            failures.append(f"retrieval missed {source}")

    # citations: only labels [1]..[k] that were actually supplied
    labels = [int(n) for n in re.findall(r"\[(\d+)\]", answer or "")]
    invalid = [n for n in labels if not 1 <= n <= len(results)]
    if invalid:
        failures.append(f"invalid citation labels {invalid}")

    if case.get("expect_insufficient"):
        if not any(phrase in answer_lower for phrase in INSUFFICIENT_PHRASES):
            failures.append("did not say the evidence is insufficient")
    else:
        if not labels:
            failures.append("no citations")
        # spaces are ignored, so "12 %" still matches "12%"
        answer_compact = re.sub(r"\s+", "", answer_lower)
        for group in case.get("must_include", []):
            if not any(re.sub(r"\s+", "", option.lower()) in answer_compact for option in group):
                failures.append(f"answer missing one of {group}")

    return failures

def run_case(client, db, case):
    history = []

    # earlier turns build the conversation, only the last turn is checked
    for question in case["turns"][:-1]:
        _, _, _, answer = ask(client, db, question, history)
        history = add_to_history(history, question, answer)

    search_query, results, _, answer = ask(client, db, case["turns"][-1], history)

    return search_query, results, answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the test questions through the RAG pipeline and check the answers.")
    parser.add_argument("--case", help="run only the test case with this id")
    parser.add_argument("--verbose", action="store_true", help="print every answer, not only failures")
    args = parser.parse_args()

    # answers can contain characters the Windows console encoding can't print
    sys.stdout.reconfigure(encoding="utf-8")

    cases = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]

    client = make_client()
    db = get_vector_store()

    passed = 0
    for case in cases:
        search_query, results, answer = run_case(client, db, case)
        failures = check_case(case, results, answer)

        if not failures:
            passed += 1

        print(f"{'PASS' if not failures else 'FAIL'}  {case['id']}")

        if failures or args.verbose:
            for reason in failures:
                print(f"      - {reason}")
            print(f"      search query: {search_query}")
            print(f"      answer: {answer}\n")

    print(f"\n{passed}/{len(cases)} passed")
