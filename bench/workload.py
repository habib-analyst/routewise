"""The fixed RouteWise benchmark workload: 20 requests across 4 task types.

Each item carries the quality tier a client would plausibly request for
it and a mechanical quality check (the same style as QuantLab's 6-prompt
factual spot check: does the answer contain the fact / structure it must
contain?). Checks are deliberately simple and are run identically for
every strategy, so the pass rate is a fair quality proxy, not a judge.

The same item with the same tier is sent under all three strategies
(all-small, all-large, routed); only the model field differs, so token
budgets are identical across strategies.
"""

from __future__ import annotations

PHOTOSYNTHESIS = (
    "Photosynthesis is the process by which green plants, algae, and some "
    "bacteria convert light energy into chemical energy stored in glucose. "
    "Inside chloroplasts, pigments such as chlorophyll absorb sunlight, "
    "mostly in the blue and red parts of the spectrum, which is why leaves "
    "appear green. The absorbed energy drives the splitting of water "
    "molecules, releasing oxygen as a byproduct, and powers the production "
    "of ATP and NADPH. These energy carriers then fuel the Calvin cycle, "
    "which fixes carbon dioxide from the air into organic molecules. "
    "Photosynthesis sustains nearly all life on Earth, either directly or "
    "indirectly, because almost every food chain begins with a producer "
    "capturing energy from sunlight, and it is also responsible for the "
    "oxygen in the atmosphere that aerobic organisms breathe."
)

BICYCLE = (
    "The bicycle developed through a series of designs in the nineteenth "
    "century. In 1817 Karl von Drais built a wooden machine with two "
    "wheels that the rider pushed along with their feet. Pedals were "
    "added in the 1860s, producing the velocipede, whose large front "
    "wheel allowed higher speeds but made the machine dangerous. The "
    "safety bicycle of the 1880s, with two equal wheels and a chain "
    "driving the rear wheel, is essentially the design still used today. "
    "Pneumatic tyres, patented by John Boyd Dunlop in 1888, made riding "
    "far more comfortable. The bicycle gave ordinary people cheap "
    "personal transport, influenced road building, and played a notable "
    "role in women's emancipation by changing dress codes and giving "
    "women independent mobility."
)

TIDES = (
    "Ocean tides are caused mainly by the gravitational pull of the Moon "
    "on the Earth's oceans. Because gravity weakens with distance, the "
    "side of the Earth facing the Moon is pulled more strongly than the "
    "centre, and the centre more strongly than the far side, which "
    "stretches the oceans into two bulges. As the Earth rotates, coastal "
    "places pass through these bulges and experience high tide roughly "
    "twice a day. The Sun also exerts a tidal force, about half the "
    "Moon's. When the Sun, Moon, and Earth align, at new and full moon, "
    "the forces add and produce the higher spring tides; when they are "
    "at right angles, the forces partly cancel and produce the weaker "
    "neap tides. Local geography, such as funnel-shaped bays, can "
    "greatly amplify the tidal range."
)

PRINTING = (
    "The printing press transformed the production of books in Europe. "
    "Before its invention, every copy of a book had to be written out by "
    "hand, a process so slow and expensive that books were rare luxury "
    "objects owned mainly by churches, universities, and wealthy nobles. "
    "Around 1440, Johannes Gutenberg of Mainz combined movable metal "
    "type, oil-based ink, and a screw press adapted from wine making "
    "into a system that could print hundreds of pages a day. His Bible "
    "of around 1455 demonstrated the quality that the new method could "
    "achieve. Printing spread rapidly: by 1500 there were presses in "
    "hundreds of European towns and an estimated ten million printed "
    "books in circulation, more than all the manuscripts produced in the "
    "previous thousand years. The fall in the price of books widened "
    "access to literacy, standardised texts so that scholars in "
    "different countries could cite the same page, and accelerated the "
    "spread of new ideas. Historians credit the press with helping to "
    "fuel the Renaissance, the Reformation, and the Scientific "
    "Revolution, because arguments, data, and criticism could now travel "
    "further and faster than ever before. Governments and churches "
    "responded with censorship and licensing systems, an early sign that "
    "whoever controls the means of reproduction shapes public debate. "
    "The press also created new trades, from compositors and proof "
    "readers to publishers and booksellers, and laid the foundation for "
    "newspapers, journals, and eventually the modern publishing "
    "industry."
)


def _item(id_, category, tier, prompt, check):
    return {
        "id": id_,
        "category": category,
        "tier": tier,
        "messages": [{"role": "user", "content": prompt}],
        "check": check,
    }


def _any_of(*needles):
    return {"type": "any_of", "needles": [n.lower() for n in needles]}


def _all_of(*needles):
    return {"type": "all_of", "needles": [n.lower() for n in needles]}


def _summary_check(*needles):
    return {
        "type": "summary",
        "needles": [n.lower() for n in needles],
        "max_output_to_input_words": 0.6,
    }


WORKLOAD = [
    # --- factual (the QuantLab-style 6-prompt spot check, expanded set) ---
    _item("F1", "factual", "balanced",
          "What is the capital of France?", _any_of("paris")),
    _item("F2", "factual", "economy",
          "Which planet is known as the Red Planet?", _any_of("mars")),
    _item("F3", "factual", "balanced",
          "What is the chemical symbol for gold?", _any_of("au")),
    _item("F4", "factual", "balanced",
          "How many sides does a hexagon have?", _any_of("six", "6")),
    _item("F5", "factual", "economy",
          "Who wrote the play Romeo and Juliet?", _any_of("shakespeare")),
    _item("F6", "factual", "balanced",
          "What is the largest ocean on Earth?", _any_of("pacific")),
    # --- summarisation -----------------------------------------------------
    _item("S1", "summary", "balanced",
          "Summarize the following text in two sentences.\n\n" + PHOTOSYNTHESIS,
          _summary_check("photosynthesis", "light", "energy")),
    _item("S2", "summary", "balanced",
          "Summarize the following text in two sentences.\n\n" + BICYCLE,
          _summary_check("bicycle", "wheel", "transport")),
    _item("S3", "summary", "balanced",
          "Summarize the following text in two sentences.\n\n" + TIDES,
          _summary_check("moon", "gravity", "tide")),
    _item("S4", "summary", "balanced",
          "Summarize the following text in three sentences.\n\n" + PRINTING,
          _summary_check("printing", "press", "books")),
    # --- code ---------------------------------------------------------------
    _item("C1", "code", "balanced",
          "Write a Python function that returns the nth Fibonacci number.",
          _all_of("def ", "return")),
    _item("C2", "code", "balanced",
          "Write Python code to check whether a string is a palindrome.",
          _any_of("def ", "[::-1]")),
    _item("C3", "code", "balanced",
          "Write a Python function that counts the vowels in a string.",
          _all_of("def ", "return")),
    _item("C4", "code", "premium",
          "Implement a Python function to compute the factorial of a "
          "number using recursion.",
          _all_of("def ", "return", "factorial")),
    _item("C5", "code", "balanced",
          "Write Python code to sort a list of dictionaries by the "
          "'age' key.",
          _all_of("sort", "key")),
    # --- reasoning ------------------------------------------------------------
    _item("R1", "reasoning", "balanced",
          "A train travels 60 km in 45 minutes. What is its average speed "
          "in km/h? Think step by step.",
          _any_of("80")),
    _item("R2", "reasoning", "balanced",
          "If every Bloops is a Razzle and every Razzle is a Lazzle, is "
          "every Bloops a Lazzle? Explain the logic.",
          _any_of("yes")),
    _item("R3", "reasoning", "balanced",
          "Sara has 3 boxes with 12 pencils each. She gives away 9 "
          "pencils. How many pencils does she have left? Calculate step "
          "by step.",
          _any_of("27")),
    _item("R4", "reasoning", "premium",
          "A fair coin is flipped 3 times. What is the probability of "
          "getting exactly 2 heads? Solve step by step.",
          _any_of("3/8", "0.375", "37.5")),
    _item("R5", "reasoning", "balanced",
          "If it takes 5 machines 5 minutes to make 5 widgets, how long "
          "would it take 100 machines to make 100 widgets? Reason step "
          "by step.",
          _any_of("5 minutes", "five minutes", "5 min")),
]


def evaluate(item: dict, output_text: str) -> bool:
    """Run the item's mechanical quality check against a model output."""
    check = item["check"]
    text = output_text.lower()
    needles = check["needles"]
    if check["type"] == "any_of":
        return any(n in text for n in needles)
    if check["type"] == "all_of":
        return all(n in text for n in needles)
    if check["type"] == "summary":
        prompt_words = len(item["messages"][0]["content"].split())
        output_words = len(output_text.split())
        short_enough = output_words <= prompt_words * check[
            "max_output_to_input_words"
        ]
        return short_enough and any(n in text for n in needles)
    raise ValueError(f"unknown check type: {check['type']}")
