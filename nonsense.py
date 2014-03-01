#!/usr/bin/env python
# encoding: utf-8
#
# Reads input and generates nonsense based on it.
#
# Usage:
# ./nonsense.py [--lookback N] [--startword Hello] [--min-length 20] [--max-length 300] [--no-cache] [--only-cache] input_file_with_lotsa_words.txt
#
# Has quite good performance compared to last version. With the swedish bible
# it needs about 20MB RAM and can create new sentences in ~0.1s with a prebuilt cache.
#
# Initially created by Håkan Waara (hwaara@gmail.com) 14 October, 2012.
#
# Do what you want with it, but please give credit and contribute improvements!

import sqlite3
import sys
import random
import datetime
import re
from collections import Counter


def stderr(str):
    sys.stderr.write(str + "\n")


def simple_time_diff(str, d1, d2):
    seconds = (d2-d1).seconds
    stderr("Execution for %s took %d second%s" % (str, seconds, ("s" if seconds > 1 else "")))


class MarkovChain(object):
    def __init__(self, input_file=None, lookback=3, no_cache=False):
        self.WORD_RE = re.compile(r"([\w\.\!\,]+)", re.UNICODE)
        self.NO_REAL_WORD_RE = re.compile(r"^[\d\.\,\:\;]*$")
        self.lookback = lookback

        if no_cache:
            self.conn = sqlite3.connect(":memory:")
        else:
            self.conn = sqlite3.connect("%s.markovdb~%d" % (input_file, lookback))

        self.c = self.conn.cursor()

        if input_file:
            self.input(input_file)

    def input(self, input_file):
        # check if cached db already exists
        if len(self.c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='markov_chain'").fetchall()) < 1:
            # re-generate everything from scratch.
            self.init_database()

            with open(input_file) as file:
                self.generate_markov_chain(file.read().decode("utf-8"))

    def init_database(self):
        self.c.execute("CREATE TABLE IF NOT EXISTS markov_chain (prefix text, suffix text, num_occurrences integer DEFAULT 0, probability real)")
        self.c.execute("CREATE INDEX prefix_index ON markov_chain (prefix)")

    def generate_markov_chain(self, input):
        d1 = datetime.datetime.now()

        # this counter dict will keep track of all the words/prefixes
        markov_chain_temp = Counter()

        # estimate num words so we can give progress
        word_count_estimate = input.count(' ') + 1

        prev_prefixes = []
        for counter, match in enumerate(re.finditer(self.WORD_RE, input)):
            word = match.groups()[0].lower()
            if not re.match(self.NO_REAL_WORD_RE, word):
                if counter % int(word_count_estimate/10.0) == 0:
                    stderr("%d%%" % round(counter/float(word_count_estimate) * 100.0))

                # add all prefixes => this word tuples
                for prefix in prev_prefixes:
                    markov_chain_temp[(prefix, word)] += 1

                # new meaning-prefix (if last word ended in dot)
                if prev_prefixes and prev_prefixes[-1][-1] == ".":
                    markov_chain_temp[("^", word)] += 1

                # generate new prefixes
                if len(prev_prefixes) >= self.lookback:
                    # remove longest prefix and add current word to all of the remaining prefixes
                    prev_prefixes.pop(0)

                prev_prefixes = [prefix + " " + word for prefix in prev_prefixes]
                prev_prefixes.append(word)

        d2 = datetime.datetime.now()
        simple_time_diff("Word indexing", d1, d2)

        # Now we have a huge markov_chain_temp variable looking like this:
        # Counter({
        #     (u'of', u'the'): 7236,
        #     (u'in', u'the'): 4379,
        #     ...
        # })

        stderr("Counting number of word occurrences...")

        # count num occurences of each prefix and put in another counter dict
        total_prefix_occurrences = Counter()
        for (prefix, suffix), count in markov_chain_temp.items():
            total_prefix_occurrences[prefix] += count

        # insert all rows from the original counter dict into the final table,
        # but collapsed into one row per prefix+suffix combo, with its count in
        # as num_occurrences and probability calculated from both counter dicts
        self.c.executemany("INSERT INTO markov_chain (prefix, suffix, num_occurrences, probability)\
            VALUES (?, ?, ?, ?)", [(prefix, suffix, count,
                                   ((1.0*count)/total_prefix_occurrences[prefix])  # probability
                                    ) for (prefix, suffix), count in markov_chain_temp.items()])

        self.conn.commit()

        d3 = datetime.datetime.now()
        simple_time_diff("Word counting", d2, d3)

        # vacuum database to keep disk usage to a minimum
        self.c.execute("VACUUM")
        d4 = datetime.datetime.now()

        simple_time_diff("Vacuuming", d3, d4)

    def choose_next_word(self, from_prefix):
        random_choice = random.random()
        i = 0
        for row in self.c.execute("SELECT suffix, probability FROM markov_chain WHERE prefix=? ORDER BY RANDOM()", (from_prefix,)):
            i += row[1]
            if i >= random_choice:
                return row[0]

    def generate_sentence(self, start_word=None, min_words=5, max_length=140, prevent_recursion=False):
        first_word = None
        if start_word:
            first_word = self.choose_next_word(start_word.lower())

        if first_word:
            word_queue = [start_word, first_word]
            out = start_word.capitalize() + " " + first_word
        else:
            first_word = self.choose_next_word(start_word or "^")
            word_queue = [first_word]
            out = first_word.capitalize()

        while word_queue[-1].find(".") == -1:
            suggestion = None

            # randomize how many words prefix we should start trying at
            for num_words_to_try in reversed(range(1, random.randint(1, self.lookback) + 1)):
                if len(word_queue) >= num_words_to_try:
                    prefix = " ".join(word_queue[-num_words_to_try:])
                    suggestion = self.choose_next_word(prefix)
                    if suggestion:
                        break

            if not suggestion:
                break

            word_queue.append(suggestion)

        out = " ".join(word_queue)

        if len(word_queue) < min_words or len(out) > max_length and not prevent_recursion:
            # hack: re-try on average 10 times if we have bad outputs :)
            stderr("Output not ideal... retrying with another run. (max 10 times)")
            out = self.generate_sentence(prevent_recursion=(random.random() * 10 < 1))

        return out

if __name__ == "__main__":
    if len(sys.argv) < 2:
        stderr("Usage: ./nonsense.py [--lookback N] [--startword Hello] [--min-length 20] [--max-length 300] [--no-cache] [--only-cache] input_file_with_lotsa_words.txt")
        sys.exit(1)

    # default arguments
    start_word = None
    max_length = 140
    lookback = 3
    no_cache = False
    only_cache = False

    # ugly but simple enough argument handling
    args = sys.argv[1:]

    if "--startword" in args:
        i = args.index("--startword")
        start_word = args[i+1].decode("utf-8")
        del args[i+1]
        del args[i]
    if "--min-length" in args:
        raise Exception("--min-length is not implemented yet!")
        i = args.index("--min-length")
        min_length = args[i+1]
        del args[i+1]
        del args[i]
    if "--max-length" in args:
        i = args.index("--max-length")
        max_length = args[i+1]
        del args[i+1]
        del args[i]
    if "--lookback" in args:
        i = args.index("--lookback")
        lookback = args[i+1]
        del args[i+1]
    if "--no-cache" in args:
        i = args.index("--no-cache")
        no_cache = True
        del args[i]
    if "--only-cache" in args:
        i = args.index("--only-cache")
        only_cache = True
        del args[i]

    input_sources = args
    source = args[0]

    stderr("Generating Markov chain for input %s..." % source)
    markov_chain = MarkovChain(input_file=args[0], lookback=lookback, no_cache=no_cache)

    if only_cache:
        stderr("Told to only generate a cache, so exiting now.")
        sys.exit(0)

    stderr("Generating sentence...")
    print markov_chain.generate_sentence(start_word=start_word, max_length=max_length).encode("utf-8")
