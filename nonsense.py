#!/usr/bin/env python
# encoding: utf-8
#
# Reads input and generates nonsense based on it.
#
# Usage:
# ./nonsense.py [--lookback N] [--startword Hello] [--min-length 20] [--max-length 300] [--no-cache] [--only-cache] [--db-connect-string dialect+driver://username:password@host:port/database] input_file_with_lotsa_words.txt
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

from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, Float, Index
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import select, update
from sqlalchemy.sql.expression import func


# Set to True to get a lot of useful debug output from sqlalchemy
VERBOSE = False

WRITE_TO_DB_AFTER_NUM_WORDS = 5000

TABLE_NAME = 'markov_chain'
TEMP_TABLE_NAME = '%s_temp' % TABLE_NAME


def stderr(str):
    sys.stderr.write(str + "\n")


def simple_time_diff(str, d1, d2):
    seconds = (d2-d1).seconds
    stderr("Execution for %s took %d second%s" % (str, seconds, ('s' if seconds > 1 else '')))


class MarkovChain(object):
    def __init__(self, input_file=None, lookback=3, no_cache=False, db_connect_string=None):
        self.WORD_RE = re.compile(r"([\w\.\!\,]+)", re.UNICODE)
        self.NO_REAL_WORD_RE = re.compile(r"^[\d\.\,\:\;]*$")
        self.lookback = lookback

        if db_connect_string:
            self.engine = create_engine(db_connect_string, echo=VERBOSE)
        else:
            if no_cache:
                self.engine = create_engine('sqlite://', echo=VERBOSE)  # :memory:
            else:
                self.engine = create_engine('sqlite:///%s.markovdb~%d' % (input_file, lookback), echo=VERBOSE)

        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        if input_file:
            self.input(input_file)

    def input(self, input_file):
        metadata = MetaData()
        metadata.bind = self.engine

        # Define the main table.
        self.table = Table(TABLE_NAME, metadata,
                           Column('prefix', String),
                           Column('suffix', String),
                           Column('num_occurrences', Integer, default=0),
                           Column('probability', Float))
        Index('prefix_index', self.table.c.prefix)

        if self.table.exists() and self.table.count().scalar() > 0:
            stderr('Table %s already exists, will reuse its data.' % TABLE_NAME)
        else:
            self.table.drop(checkfirst=True)
            metadata.create_all()
            with open(input_file) as file:
                self.generate_markov_chain(file.read().decode("utf-8"))

    def generate_markov_chain(self, input):
        # create "temporary" table for initial data batch
        metadata = MetaData()
        temptable = Table('%s' % TEMP_TABLE_NAME, metadata,
                          Column('prefix', String),
                          Column('suffix', String))
        Index('prefix_suffix_index', temptable.c.prefix, temptable.c.suffix)
        temptable.drop(self.engine, checkfirst=True)
        metadata.create_all(self.engine)

        # interface to the db
        def save_suffixes(suffixes):
            if suffixes:
                self.session.execute(temptable.insert(), [{
                    'prefix': prefix,
                    'suffix': suffix,
                } for prefix, suffix in suffixes])
                self.session.commit()

        d1 = datetime.datetime.now()

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

        d2 = datetime.datetime.now()
        simple_time_diff("Word indexing", d1, d2)

        stderr("Counting number of word occurrences...")

        # insert all rows from the temporary table into the final table, but
        # collapsed into one row per prefix+suffix combo, with its count in the
        # as num_occurrences in the final table
        rows = self.session.execute(select([
            temptable.c.prefix,
            temptable.c.suffix,
            func.count()
        ]).group_by(temptable.c.prefix, temptable.c.suffix)).fetchall()

        self.session.execute(self.table.insert(), [{
            'prefix': row[0],
            'suffix': row[1],
            'num_occurrences': row[2]
        } for row in rows])

        self.session.commit()

        temptable.drop(self.engine, checkfirst=True)

        self.session.commit()

        d3 = datetime.datetime.now()
        simple_time_diff("Word counting", d2, d3)

        stderr("Calculating probabilities...")

        # re-count from num occurences of each suffix => probability (from 0.0 - 1.0)
        rows = self.session.execute(select([self.table.c.prefix, func.sum(self.table.c.num_occurrences)]).group_by(self.table.c.prefix)).fetchall()
        for row in rows:
            self.session.execute(update(self.table).values(
                probability=(1.0*self.table.c.num_occurrences)/float(row[1])
            ).where(self.table.c.prefix == row[0]))

        self.session.commit()

        # vacuum database to keep disk usage to a minimum
        # TODO: This is not database independent; can we check for vacuum support?
        #self.session.execute("VACUUM")
        # TODO: Fix error "VACUUM cannot run inside a transaction block" on postgres

        self.session.commit()

        d4 = datetime.datetime.now()

        simple_time_diff("Probabilities and vacuuming", d3, d4)

    def choose_next_word(self, from_prefix):
        random_choice = random.random()
        i = 0
        for row in self.session.execute(select([
            self.table.c.suffix,
            self.table.c.probability
        ]).where(self.table.c.prefix == from_prefix).order_by(func.random())):  # TODO: 'random()' is not database independent
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
        stderr("Usage: ./nonsense.py [--lookback N] [--startword Hello] [--min-length 20] [--max-length 300] [--no-cache] [--only-cache] [--db-connect-string dialect+driver://username:password@host:port/database] input_file_with_lotsa_words.txt")
        sys.exit(1)

    # default arguments
    start_word = None
    max_length = 140
    lookback = 3
    no_cache = False
    only_cache = False
    db_connect_string = None

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
    if "--db-connect-string" in args:
        i = args.index("--db-connect-string")
        db_connect_string = args[i+1]
        del args[i+1]
        del args[i]

    input_sources = args
    source = args[0]

    stderr("Generating Markov chain for input %s..." % source)
    markov_chain = MarkovChain(input_file=args[0], lookback=lookback, no_cache=no_cache, db_connect_string=db_connect_string)

    if only_cache:
        stderr("Told to only generate a cache, so exiting now.")
        sys.exit(0)

    stderr("Generating sentence...")
    print markov_chain.generate_sentence(start_word=start_word, max_length=max_length).encode("utf-8")
