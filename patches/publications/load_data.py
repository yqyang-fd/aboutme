# %% load data
from collections import defaultdict
import os


# %%
def get_venue(entry: dict):
    if "journal" in entry:
        return entry["journal"]
    elif "booktitle" in entry:
        return entry["booktitle"]
    else:
        return ""


def format_entry(entry: dict):
    title = entry["title"]
    authors = entry["author"]
    venue = get_venue(entry)
    url = entry.get("url", "")
    s = f"""- **{title}** [link]({url})</br>*{venue}*</br>{authors}\n\n"""
    return s


def write_md(papers: dict[int, list[dict]], target: str):
    years = sorted(papers.keys(), reverse=True)

    with open(target, "w", encoding="utf-8") as f:
        f.write("# Publications\n\n")
        for year in years:
            f.write(f"## {year}\n\n")
            for entry in papers[year]:
                f.write(format_entry(entry))
            f.write("\n")
            print(f"Write {len(papers[year])} entries in {year} to {target}.")


def load_bib_file(bibfile: str) -> dict[int, list[dict]]:
    import bibtexparser

    with open(bibfile, encoding="utf-8") as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)

    papers = defaultdict(list)

    print(f"Processing {len(bib_database.entries)} entries in {bibfile}...")

    for entry in bib_database.entries:
        year = int(entry["year"])
        papers[year].append(entry)
    return papers


# %%


def load_msr_html(html_file: str) -> dict[int, list[dict]]:
    from bs4 import BeautifulSoup

    with open(html_file, encoding="utf-8") as f:
        html = f.read()
        # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    papers = defaultdict(list)

    # view by year is the first <ul> tag with class "accordion"
    # each year is a <h4> tag
    # each paper is an <article> tag

    block = soup.find("ul", class_="accordion")
    years = [year.get_text().strip() for year in block.find_all("h4")]
    years = [int(year) for year in years if year.isnumeric()]

    for year in years:
        articles = block.find(
            "div", attrs={"id": f"collapse-group-year_filter-publications-1-{year}"}
        ).find_all("article")
        print(year, len(articles))
        for article in articles:
            title = article.find("h5").find("a")
            authors = article.find("p", class_="base m-0 content-excerpt__people")
            venue = article.find(
                "p", class_="base m-0 content-excerpt__journal-and-issue"
            )
            papers[int(year)].append(
                {
                    "title": title.get_text().strip(),
                    "author": authors.get_text().strip(),
                    "journal": venue.get_text().strip() if venue else "",
                    "year": year,
                    "url": title["href"],
                }
            )
    return papers


def correct_publications(papers: dict[int, list[dict]], correct_file: str):
    import json

    with open(correct_file, encoding="utf-8") as f:
        corrections = json.load(f)

    for year, entries in papers.items():
        for entry in entries:
            title = entry["title"]
            if title in corrections:
                entry.update(corrections[title])
    return papers


# %%

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help=f"load data from *.bib or *.html")
    parser.add_argument(
        "--output", "-o", type=str, default="../../docs/publications.md"
    )
    parser.add_argument(
        "--correct",
        "-c",
        type=str,
        default=None,
        help="correct the publication from the given *.json file",
    )
    args = parser.parse_args()

    if not args.input.endswith(".bib") and not args.input.endswith(".html"):
        raise ValueError(f"Unsupported file format: {args.input}")

    papers = (
        load_bib_file(args.input)
        if args.input.endswith(".bib")
        else load_msr_html(args.input)
    )
    if args.correct:
        papers = correct_publications(papers, args.correct)
    write_md(papers, args.output)
