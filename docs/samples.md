# Text samples

Real samples from the checkpoints on the Hub. They are not chosen: each is the
output of `generate.py` with the settings below, first run.

Settings for all samples: `--temperature 0.8 --top_k 50 --max_new_tokens 120`.
Sample *i* uses `torch.manual_seed(seed + i)` with `--seed 0`. Device: Apple MPS.
Generation stops early when the model writes the end-of-text token.

Command:

```bash
python generate.py <repo id> --prompt "<prompt>" --num_samples 2   # TinyStories; WikiText-103 used 1
```

## TinyStories (Phase 1, d_model 384, seed 42)

Prompt: `Once upon a time, there was a little dog named`

### `sbartlett97/sot-phase1-tinystories-dense-seed42` (dense baseline)

```text
--- sample 0 ---
Once upon a time, there was a little dog named Spot. Spot was very eager to learn new things. He wanted to go to the mall to see all the fun things there. Spot had a good idea. He wanted to go to the mall and play with his friend, the cat.

Spot and the cat went to the mall. They saw a big, shiny toy car. Spot was very excited to play with the toy car. He played with the car all day long. But after a while, Spot started to feel a little scared. He didn't want anyone to find him there.

Finally, Spot found a nice,
--- sample 1 ---
Once upon a time, there was a little dog named Max. Max loved to play in the park with his friends. They would run and jump and have so much fun.

One day, Max met a little girl named Lily. Lily was sad because she lost her favorite toy. Max said, "Don't worry, I will help you find it." Max barked happily and barked, "Thank you!"

Max barked happily and said, "You're welcome!" They walked together through the park and found Lily's toy. Lily was so happy to have her toy back.

Max and Lily played together all day.
```

### `sbartlett97/sot-phase1-tinystories-sot-seed42` (gated supernet)

```text
--- sample 0 ---
Once upon a time, there was a little dog named Spot. Spot was very eager to learn new things. He loved to play and run around the house with his owner.

One day, Spot's owner mentioned that Spot had a new job. It was called a lab. Spot was happy to hear about it. He went inside and played with his new home.

After Spot played with his new house, he started to feel bored. He wanted to play with his new friends. Spot's owner told him it was time to go home and Spot went to sleep.

Spot went back to his owner and went to sleep. When
--- sample 1 ---
Once upon a time, there was a little dog named Max. Max loved to play in the park with his friends. They would run and jump and have so much fun.

One day, Max's friend, a cat named Lily, said, "Max, I recommend that you come over here and play. It's important to be loyal to me." Max felt sad and said, "I'm sorry, Lily. I don't want to play with you."

Lily thought for a moment and then said, "Max, you may want to play with me and my friends. I will always remember to be loyal to you."
```

### `sbartlett97/sot-phase1-tinystories-discovered_retrained-seed42` (discovered architecture, trained from scratch)

```text
--- sample 0 ---
Once upon a time, there was a little dog named Spot. Spot was very eager to learn new things. One day, Spot saw a cat in the park. Spot wanted to help the cat.

Spot went to the park and started to play with the cat. The cat was not happy. Spot did not know what to do. The cat jumped on Spot and tried to bite him. Spot was scared and ran away.

Spot's owner saw what happened and came to help. She took Spot to the vet. The vet checked his paw and found out that Spot had a hole in his paw. The vet was very nice and put
--- sample 1 ---
Once upon a time, there was a little dog named Max. Max loved to play in the park with his friends. They would run and jump and have so much fun.

One day, Max met a little girl named Lily. Lily was sad because she lost her favorite toy. Max said, "Don't worry, I will help you find it." Max barked and wagged his tail.

Max and Lily decided to look for the toy. They searched and searched until they found the toy under a tree. Lily was so happy and said, "Thank you, Max! You are a good friend."

Max wagged
```

**Notes**

- All three models write fluent, grammatical TinyStories prose with named
  characters, dialogue and paragraphs.
- The plot is weak in all three. Characters change goals, and sentences repeat
  ("Max barked happily and barked"). This is normal at this size.
- With the same seed, the three models often start with the same words
  ("Spot was very eager to learn new things"). The dataset is very
  repetitive, and the models share the tokenizer and data order.
- We see no clear quality difference between the three from these samples.
  The perplexity difference (seed 42: dense 4.63, sot 4.75, retrained 4.89; corrected values, RESEARCH_LOG.md 2026-10-06) is too
  small to show in a few samples.

## WikiText-103 (d_model 768, seed 42)

### `sbartlett97/sot-phase1-wikitext103-dense-seed42-tc768d013` (dense baseline)

Prompt: `The history of the city begins in`

```text
--- sample 0 ---
The history of the city begins in the 1st century , with its name from the Roman era . The walls of the citadel , which was built on the site of the old Roman fort , were built by St. John the Baptist . The Roman defences were initially based on defensive , and were constructed of stone and brick , with an iron dome supported on a slate roof . Later , a wall of Roman and Byzantine architecture was built , with a large stone roof supported in gold . The walls were built with cast iron columns , but this was changed by the addition of an artillery battery . As the Roman fortifications were further expanded , the
```

Prompt: `The species is a medium-sized bird found in`

```text
--- sample 0 ---
The species is a medium-sized bird found in many areas of the world , including the British Isles and the United Kingdom . It is a medium @-@ sized common name for the boggy ground , and is found in all regions except for the southern parts .
```

### `sbartlett97/sot-phase2-wikitext103-heuristic-seed42-tc768d013-v8-tailfreeze` (Phase 2 attempt 8)

Prompt: `The history of the city begins in`

```text
--- sample 0 ---
The history of the city begins in the city of Kraków . In the 1920s , the city was home to the Kraków @-@ Croatian Sultanate of Croatia , who built a fort there . The city is divided into two parts , the central and eastern parts .
```

Prompt: `The species is a medium-sized bird found in`

```text
--- sample 0 ---
The species is a medium-sized bird found in the tropical waters of the Great Barrier Reef . The range is known to be similar to that of some other small @-@ bodied species , but there is insufficient evidence for a close relationship between them .
```

**Notes**

- Both models write in the Wikipedia style of the dataset, with its
  tokenisation (` , `, ` . `, `@-@` for a hyphen).
- The text is fluent but the facts are wrong ("walls built by St. John the
  Baptist", "the Kraków-Croatian Sultanate of Croatia", a bird in "tropical
  waters of the Great Barrier Reef"). A 150M-parameter model trained for
  10,000 steps does not store facts reliably.
- The attempt-8 checkpoint predates the zero-effect change. It loads with
  every scale at 1, so this sample comes from the full supernet, with its
  dormant and dead units still adding to the output. It is not a sample of
  the present structure alone (see `RESEARCH_LOG.md`, 2026-10-14 and
  2026-10-15). Loading prints a "newly initialized" warning for the scale and
  probe buffers. This is expected: they take their default values.

### Attempt 8, present structure only

Same checkpoint and settings, but every unit that is not present at step
10000 has scale 0 (9 layers, 42 heads, 446 FFN groups remain). This is the
structure that `compute.py` counts as 0.466 of dense FLOPs, and that
`structure_eval.py` scores at 132 val ppl. Command:
`python results/compute/present_only_sample.py`.

```text
--- 'The history of the city begins in' ---
The history of the city begins in the the opening gate , is now a a long long , and the the the the that that it opened , and the " " " " " " " " ( ) " " " " " " " " " " " " " " " " " " " " " " " " , " " " " " " " " " " " " " " " " " " " " , " " " " " " " " " " " " " , " " " " " " " " " " " " . . " " " " " " " " " " , ,
--- 'The species is a medium-sized bird found in' ---
The species is a medium-sized bird found in some areas areas , and is often found in the Suluuluuluuluuluuluuluuluuluuluuluuluuluuluuluuluuluulu[... the same 3 characters repeat to the end]
```

- The present structure alone breaks down after a few words. It repeats
  tokens and loops. The fluent text above came from the dormant units,
  not from the structure the controller selected.
- This is why the code now gives dormant and dead units exactly zero
  effect during training (`RESEARCH_LOG.md`, 2026-10-15). A model trained
  that way must produce its text with the present structure alone.

### `sbartlett97/sot-phase2-wikitext103-heuristic-seed42-tc768d013-v9-zeroeffect` (Phase 2 attempt 9)

Attempt 9 trained on the zero-effect code, so inactive units have no
effect. These samples come from the present structure only: 8 layers, 36
heads and 357 FFN groups, 0.425 of dense FLOPs/token, 30.61 val ppl.
Same settings as attempt 8 above, with `generate.py` and no changes.

Prompt: `The history of the city begins in`

```text
--- sample 0 ---
The history of the city begins in the city of the present @-@ day city of Poughkeepsie ( modern @-@ day Washington ) . The city of Poughkeepsie was one of the largest city residents in the United States . Poughkeepsie had previously been the seat of William McKinley High School , and the city of Poughkeepsie was also the city 's major community .
```

Prompt: `The species is a medium-sized bird found in`

```text
--- sample 0 ---
The species is a medium-sized bird found in the tropical waters of the Great Barrier Reef area , and is known to fly over water as the " Blue Mountains " . The eastern polar bears are usually found in dry , dry , dry , dry and dry environments . The westernmostmost species is a greyish blue @-@ pink colour , with a brown color on the body . The northern polar bears are also found in the eastern polar forms , and are often found in the polar bear . The southern polar bears are also found in polar bears and polar bears .
```

- Attempt 8's present structure loops after a few words. Attempt 9's
  present structure, with less compute, writes full sentences in the
  dataset's style. The zero-effect training made the selected structure
  usable on its own.
- It is weaker than the dense model and the attempt-8 supernet. It
  repeats words more ("dry , dry , dry", "polar bears and polar bears")
  and drifts off the topic: the bird becomes polar bears. This agrees
  with its higher val ppl (30.61, against 24.30 for dense 768).
- With the same seed, attempt 9 starts the bird sample like attempt 8
  ("tropical waters of the Great Barrier Reef"). The two runs share the
  tokenizer, data order and seed.

## Just for fun

More prompts, one run each, not chosen. Settings: `--seed 1
--max_new_tokens 90 --temperature 0.8 --top_k 50`.

### `sbartlett97/sot-phase1-tinystories-dense-seed42`

```text
Once upon a time, a dragon wanted to learn how to bake a cake. He was very competitive and wanted to help all the other animals. They all had yummy food and were having so much fun.

But then, a little bird came and asked for help. The dragon said, "Don't worry! I can help you bake the cake. We can bake it together!" So, the dragon and the little bird worked very hard and baked the cake. It was the best cake ever!

The dragon
```

```text
Tom found a robot in the garden. The robot said, "Hello, Tom! Do you want to play with me?" They played together and had fun. They had a lot of fun.

But then, a big cat came. The cat was hungry and wanted to eat them. The mouse was scared and ran away. Tom and Sam were sad and did not know what to do. Tom saw the cat and felt sorry for him.

Tom said, "Don't be sad, cat
```

```text
The little cat looked at the moon and said, "Thank you, moon." The moon smiled and said, "You're welcome, little cat. Always remember that." The little cat purred and went back to sleep.
```

### `sbartlett97/sot-phase1-wikitext103-dense-seed42-tc768d013`

```text
The Transformer is a deep learning architecture that can be seen as a kind of " virtual house " . It also features an open @-@ ended multiplayer mode with a single player choice based on a race . The game was released in the UK on 18 January 2012 , and in the United States on 1 March 2012 . It was released in the UK on 21 December 2012 . A port of the game was released on 12 March 2012 on the PC , and a Japanese release was given after the game
```

```text
In 2026 , a team of researchers discovered that the body may be a remnant of a giant giant , possibly a black hole .
```

```text
The cat ( Felis catus ) is one of the seven cheetae on the left , the others are in the centre of the circle . The cheetae on the right are very similar , but are not directly connected and are parallel with one another . Also the two cheetae on the right are not connected by curved joints as in the cheetae . The cheetae are very similar . A study of the cheetae noted that in case of the cheetae
```

### `sbartlett97/sot-phase2-wikitext103-heuristic-seed42-tc768d013-v9-zeroeffect`

```text
The Transformer is a deep learning architecture that can be seen as a learning center on an object , and the environment is an open space and learning center . The game also features a number of levels , including a variety of gameplay , and the developers have an " emotional " feel . The story allows players to choose the best and most rewarding areas as they are " able to control them " with " a very high level of play . "
```

```text
In 2026 , a team of researchers discovered that one of the last proteins on a given scale was a protein , and the team reported that the protein 's protein was a single protein . The protein 's protein 's initial size and the protein 's length was reduced by the addition of the protein 's protein into a protein rather than a protein 's protein . This would later evolve the protein 's protein 's protein on the protein 's surface .
```

```text
The cat ( Felis catus ) is one of the oldest mammals on the continent and one of the most abundant animals on the continent . It covers a range of more than 80 species of mammals , and can be found in the most part of the continent . It is the second most extinct species on Europe . It has been dated to over 20 million years , and has been collected in many parts of Europe , including parts of Europe , China , and the Middle East . It is now rare in
```

- The TinyStories model keeps the shape of a story: a problem, help, a
  happy end. It loses track of the cast ("The mouse was scared", "Tom and
  Sam").
- The WikiText-103 model knows what a Wikipedia article sounds like, but
  not what it says. "Transformer" becomes a video game with release
  dates.
- Attempt 9 also turns the Transformer into a video game. It gets stuck
  on one word more often ("the protein 's protein 's protein"), and it
  calls the cat "the second most extinct species on Europe".

## Check

For the TinyStories dense model, the loss of the model on its own 200-token
sample is 0.875 (ppl 2.40). Its loss on the first validation story is 1.40
(ppl 4.06). A sampled text at temperature 0.8 is expected to have a lower
loss than real text. Both values are finite and in the range of the reported
val ppl (4.63), so the sampling path and the evaluation path agree.
