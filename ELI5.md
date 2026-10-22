# RouteWise, explained simply

Imagine you run a help desk with two workers.

The junior worker is quick and cheap. The senior worker is slower to
book and costs more, but is better at hard problems. If every question
goes to the senior, you waste money on questions like "what is the
capital of France?" If everything goes to the junior, the hard
questions get shaky answers.

RouteWise is the person at the front desk who decides, for each
question:

1. What kind of question is this? (a fact, a summary, some code, a
   puzzle)
2. How hard does it look?
3. How much is the customer willing to spend? (that is the quality
   tier: economy, balanced, premium)
4. How long may the answer be? (that is the token budget)

Easy question, cheap tier: junior. Hard question or generous tier:
senior. And every question gets a size limit, like "your answer must
fit on one page". If someone hands in a ten-page question on the
economy tier, the front desk trims it down to size before passing it
on. If it is bigger than the building allows, it is turned away at
the door. Every decision is written in a log book, with the limit
that was set and what was actually used.

What did I learn by testing it?

- The junior got 17 out of 20 questions right, the senior 15, and the
  front-desk routing also 15, while sending 40% of the work to the
  junior. Routing worked, but on these questions the junior was
  already good, so there was little left for routing to win.
- The size limit bit back. On the puzzle questions, both workers
  liked to think out loud step by step, and the one-page limit cut
  them off before they wrote the final answer. A limit is a good
  guardrail, but set it too tight and it throws away exactly the
  answers you paid extra for.
- The sneakiest problem was not the workers at all. It was the
  building. The computers here are shared, and when the senior
  worker's books sat on the shelf too long, the building quietly
  threw the pages away to make room for other people. The next
  question then took forever while the pages were fetched back. The
  fix was to photocopy the books into the senior's own notebook,
  which the building is not allowed to throw away. In computer words:
  copy the model weights into memory the system cannot evict.
