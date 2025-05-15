
import wptools
import re
page = wptools.page("Joker")
page.get_parse()  # Can also use get() or get_wikidata()

infobox = page.data.get("infobox")
budget = infobox.get("budget")
print("Trsing")
def clean_budget(raw_budget: str) -> str:
    raw_budget = str(raw_budget)

    # Step 1: Remove ALL wikitext templates like {{...}} (including nested)
    # This uses a smarter regex to handle multiple and nested-ish cases
    cleaned = re.sub(r"\{\{[^{}]*\}\}", "", raw_budget)

    # Step 2: Remove any HTML comments or residual brackets (just in case)
    cleaned = re.sub(r"<!--.*?-->", "", cleaned)
    cleaned = re.sub(r"\[.*?\]", "", cleaned)

    # Step 3: Collapse whitespace and remove stray commas
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.replace(" ,", ",").strip(" ,\n\t")
    print(cleaned)
    return cleaned

clean_budget(budget)
# print(infobox.get('budget'))
print(infobox)