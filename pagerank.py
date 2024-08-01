import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    n = len(corpus)

    if corpus[page]:
        # probablity to access randomly
        random_pro = (1-damping_factor)/n
        # probablity to access from page
        link_pro = damping_factor/len(corpus[page])
    else:
        random_pro = 1/n
        link_pro = 0
    # to store result
    probability = {}
    for key in corpus:
        if key in corpus[page]:
            # adding both type of probalities
            probability[key] = link_pro+random_pro
        else:         
            probability[key] = random_pro
            
    return probability


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    probability = {}
    for i in corpus:
        probability[i] = transition_model(corpus, i, damping_factor)
    # first random state
    state = random.choice(list(corpus))
    ans = {}
    for i in range(n):
        # storing number of time we visit the page
        ans[state] = ans.get(state, 0) + 1
        # choosing next page on basis of probablity
        state = random.choices(list(probability[state]), list(probability[state].values()), k=1)[0]
    # converting visit count to probablity
    for key, element in ans.items():
        ans[key] = ans[key]/n
    return ans


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    n = len(corpus)
    # if there is any page with no links
    for key, joins in corpus.items():
        if not joins:
            corpus[key] = {x for x in corpus}
    links = {}
    # making new dict for storing the reverse corpus
    # it indicate by which pages we can access current page
    for key, joins in corpus.items():
        if joins:
            for i in joins:
                val = links.get(i)
                if val:
                    val.add(key)
                    links[i] = val
                else:
                    links[i] = {key}

    # final probability holder
    pr = {}
    for i in corpus:
        pr[i] = 1/n
    diff = 1
    change = True
    while change:
        change = False
        for i in corpus: 
            sumation = 0
            if i in links:
                # if there are links to that page
                for j in links[i]:
                    sumation += (pr[j]/(len(corpus[j])))
                new_value = ((1-damping_factor)/n)+(damping_factor*sumation)
                diff = abs(pr[i] - new_value)
                if diff >= 0.001:
                    change = True
                pr[i] = new_value
            else:
                # if there are no links to this page
                new_value = (1-damping_factor)/n
                diff = abs(pr[i] - new_value)
                if diff >= 0.001:
                    change = True
                pr[i] = new_value
    return pr


if __name__ == "__main__":
    main()
