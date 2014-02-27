#!/usr/bin/env python
# encoding: utf-8
#
# Reads input and generates nonsense based on it.
#
# Usage:
# ./nonsense.py [--lookback N] [--startword Hello] [--min-length 20] [--max-length 300] [--no-cache] [--only-cache] [--db sqlite|postgres dsn] input_file_with_lotsa_words.txt
#
# Has quite good performance compared to last version. With the swedish bible
# it needs about 20MB RAM and can create new sentences in ~0.1s with a prebuilt cache.
#
# Initially created by Håkan Waara (hwaara@gmail.com) 14 October, 2012.
#
# Do what you want with it, but please give credit and contribute improvements!

import sys
import random
import datetime
import re

WRITE_TO_DB_AFTER_NUM_WORDS = 5000


def stderr(str):
    sys.stderr.write(str + "\n")


def simple_time_diff(str, d1, d2):
    seconds = (d2-d1).seconds
    stderr("Execution for %s took %d second%s" % (str, seconds, ("s" if seconds > 1 else "")))


class MarkovChain(object):
    def __init__(self, input_file=None, lookback=3, no_cache=False, db="sqlite", dsn=None):
        self.WORD_RE = re.compile(r"([\w\.\!\,]+)", re.UNICODE)
        self.NO_REAL_WORD_RE = re.compile(r"^[\d\.\,\:\;]*$")
        self.lookback = lookback
        self.db = db

        if self.db == "sqlite":
            import sqlite3
            self.table_name = 'markov_chain'
            self.placeholder = '?'
            if no_cache:
                self.conn = sqlite3.connect(":memory:")
            else:
                self.conn = sqlite3.connect("%s.markovdb~%d" % (input_file, lookback))
        elif self.db == "postgres":
            import psycopg2
            # In Python 2, psycopg2 returns byte string by default, so we need
            # this to get unicode and avoid a lot of decode("utf-8")
            import psycopg2.extensions
            psycopg2.extensions.register_type(psycopg2.extensions.UNICODE)
            psycopg2.extensions.register_type(psycopg2.extensions.UNICODEARRAY)
            self.table_name = 'markov_chain_%s_%d' % (input_file.replace('.', '_'), lookback)
            self.placeholder = '%s'
            self.conn = psycopg2.connect(dsn)
        else:
            stderr("Database '%s' is not supported" % db)
            sys.exit(1)

        self.c = self.conn.cursor()

        if input_file:
            self.input(input_file)

    def input(self, input_file):
        # check if cached db already exists
        if self.db == "sqlite":
            query = "SELECT name FROM sqlite_master WHERE type='table' AND name='%s'" % self.table_name
        elif self.db == "postgres":
            query = "SELECT table_name FROM information_schema.tables WHERE table_name='%s'" % self.table_name
        self.c.execute(query)
        if not self.c.fetchone():
            # re-generate everything from scratch.
            self.init_database()

            with open(input_file) as file:
                self.generate_markov_chain(file.read().decode("utf-8"))

    def init_database(self):
        self.c.execute("CREATE TABLE IF NOT EXISTS %s (id serial primary key, prefix text, suffix text, num_occurrences integer DEFAULT 0, probability real)" % self.table_name)
        self.c.execute("CREATE INDEX %s_prefix_index ON %s (prefix)" % (self.table_name, self.table_name))
        self.conn.commit()

    def vacuum(self):
        if self.db == 'sqlite':
            self.c.execute("VACUUM")
        elif self.db == 'postgres':
            old_isolation_level = self.conn.isolation_level
            self.conn.set_isolation_level(0)
            self.c.execute("VACUUM FULL")
            self.conn.set_isolation_level(old_isolation_level)

    def generate_markov_chain(self, input):
        # interface to the db
        def save_suffixes(suffixes):
            self.c.executemany("INSERT INTO %s_temp (prefix, suffix) VALUES (%s, %s)" % (self.table_name, self.placeholder, self.placeholder), suffixes)

        d1 = datetime.datetime.now()

        # create "temporary" table for initial data batch
        self.c.execute("DROP TABLE IF EXISTS %s_temp" % self.table_name)
        self.c.execute("CREATE TABLE %s_temp (prefix text, suffix text)" % self.table_name)
        self.c.execute("CREATE INDEX prefix_suffix_index ON %s_temp (prefix, suffix)" % self.table_name)

        # estimate num words so we can give progress
        word_count_estimate = input.count(" ") + 1

        prev_prefixes = []
        unsaved_suffixes = []
        for counter, match in enumerate(re.finditer(self.WORD_RE, input)):
            word = match.groups()[0].lower()
            if not re.match(self.NO_REAL_WORD_RE, word):
                if counter % int(word_count_estimate/10.0) == 0:
                    stderr("%d%%" % round(counter/float(word_count_estimate) * 100.0))

                # add all prefixes => this word tuples
                for prefix in prev_prefixes:
                    unsaved_suffixes.append((prefix, word))

                # new meaning-prefix (if last word ended in dot)
                if prev_prefixes and prev_prefixes[-1][-1] == ".":
                    unsaved_suffixes.append(("^", word))

                # generate new prefixes
                if len(prev_prefixes) >= self.lookback:
                    # remove longest prefix and add current word to all of the remaining prefixes
                    prev_prefixes.pop(0)

                prev_prefixes = [prefix + " " + word for prefix in prev_prefixes]
                prev_prefixes.append(word)

            # flush suffixes to database
            if counter % WRITE_TO_DB_AFTER_NUM_WORDS == 0:
                save_suffixes(unsaved_suffixes)
                unsaved_suffixes = []

        # final save of suffixes to database
        save_suffixes(unsaved_suffixes)

        self.conn.commit()

        d2 = datetime.datetime.now()
        simple_time_diff("Word indexing", d1, d2)

        stderr("Counting number of word occurrences...")

        # insert all rows from the temporary table into the final table, but
        # collapsed into one row per prefix+suffix combo, with its count in the
        # as num_occurrences in the final table
        self.c.execute("INSERT INTO %s (prefix, suffix, num_occurrences)\
            SELECT prefix, suffix, count(*) FROM %s_temp GROUP BY prefix, suffix" % (self.table_name, self.table_name))

        self.c.execute("DROP TABLE %s_temp" % self.table_name)

        self.conn.commit()

        d3 = datetime.datetime.now()
        simple_time_diff("Word counting", d2, d3)

        stderr("Calculating probabilities...")

        # re-count from num occurences of each suffix => probability (from 0.0 - 1.0)
        self.c.execute("SELECT prefix, sum(num_occurrences) FROM %s GROUP BY prefix" % self.table_name)
        for row in self.c.fetchall():
            self.c.execute("UPDATE %s SET probability=((1.0*num_occurrences)/%s) WHERE prefix=%s" % (self.table_name, self.placeholder, self.placeholder), (float(row[1]), row[0]))

        self.conn.commit()

        # vacuum database to keep disk usage to a minimum
        self.vacuum()
        d4 = datetime.datetime.now()

        simple_time_diff("Probabilities and vacuuming", d3, d4)

    def choose_next_word(self, from_prefix):
        random_choice = random.random()
        i = 0
        self.c.execute("SELECT suffix, probability FROM %s WHERE prefix=%s ORDER BY RANDOM()" % (self.table_name, self.placeholder), (from_prefix,))
        for row in self.c:
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
        stderr("Usage: ./nonsense.py [--lookback N] [--startword Hello] [--min-length 20] [--max-length 300] [--no-cache] [--only-cache] [--db sqlite|postgres dsn] input_file_with_lotsa_words.txt")
        sys.exit(1)

    # default arguments
    start_word = None
    max_length = 140
    lookback = 3
    no_cache = False
    only_cache = False
    db = "sqlite"
    dsn = None

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
    if "--db" in args:
        i = args.index("--db")
        db = args[i+1]
        dsn = args[i+2]
        del args[i+2]
        del args[i+1]
        del args[i]

    input_sources = args
    source = args[0]

    stderr("Generating Markov chain for input %s..." % source)
    markov_chain = MarkovChain(input_file=args[0], lookback=lookback, no_cache=no_cache, db=db, dsn=dsn)

    if only_cache:
        stderr("Told to only generate a cache, so exiting now.")
        sys.exit(0)

    stderr("Generating sentence...")
    print markov_chain.generate_sentence(start_word=start_word, max_length=max_length).encode("utf-8")
