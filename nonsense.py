#!/usr/bin/env python
# encoding: utf-8
#
# Reads input and generates nonsense based on it.
#
# Usage: 
# ./nonsense.py [[--lookback N]|[--startword Hello]|[--min-length 20]|[--max-length 300]|--no-cache] input_file_with_lotsa_words.txt"
#
# Has quite good performance compared to last version. With the swedish bible
# it needs about 20MB RAM and can create new sentences in ~0.1s with a prebuilt cache.
#
# Initially created by Håkan Waara (hwaara@gmail.com) 14 October, 2012.
#
# Do what you want with it, but please give credit and contribute improvements!

import sqlite3
import os
import sys
import random
import re
from decimal import Decimal

def stderr(str):
    sys.stderr.write(str + "\n")

class MarkovChain(object):
    def __init__(self, input_file=None, lookback=3, no_cache=False):
        self.WORD_RE = re.compile(r"([\w\.\!\,]+)", re.UNICODE)
        self.NO_REAL_WORD_RE = re.compile(r'^[\d\.\,\:\;]*$')
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
       self.c.execute("CREATE TABLE IF NOT EXISTS markov_chain (prefix text, suffix text, num_occurences integer DEFAULT 0, probability real, UNIQUE (prefix, suffix) )");

    def generate_markov_chain(self, input):        
        # interface to the db
        def add_suffix(prefix, suffix):
            self.c.execute("INSERT OR IGNORE INTO markov_chain (prefix, suffix) VALUES (?, ?)", (prefix, suffix))
            self.c.execute("UPDATE markov_chain SET num_occurences = num_occurences + 1 WHERE prefix=? AND suffix=?", (prefix, suffix))
        
        # estimate num words so we can give progress
        word_count_estimate = len(input.split(" "))

        i=0
        prev_prefixes = []
        for match in re.finditer(self.WORD_RE, input):
            word = match.groups()[0].lower()
            if not re.match(self.NO_REAL_WORD_RE, word):
                i+=1
                if i % (word_count_estimate/10) == 0:
                    print "%d%%" % int(i/float(word_count_estimate) * 100.0)
                
                # add all prefixes => this word tuples
                for prefix in prev_prefixes:
                    add_suffix(prefix, word)
                    
                # new meaning-prefix (if last word ended in dot)
                if prev_prefixes and prev_prefixes[-1][-1] == ".":
                    add_suffix("^", word)
                    
                # generate new prefixes
                if len(prev_prefixes) >= self.lookback:
                    # remove longest prefix and add current word to all of the remaining prefixes
                    prev_prefixes.pop(0)
                
                temp = map(lambda prefix: prefix + " " + word, prev_prefixes)
                prev_prefixes = temp
                prev_prefixes.append(word)

        self.conn.commit()
        stderr("calculating probabilities…")
    
        # re-count from num occurences of each suffix => probability (from 0.0 - 1.0)
        for row in self.c.execute("SELECT prefix, sum(num_occurences) FROM markov_chain GROUP BY prefix").fetchall():
            self.c.execute("UPDATE markov_chain SET probability=((1.0*num_occurences)/?) WHERE prefix=?", (float(row[1]), row[0]))

        self.conn.commit()
            
    def choose_next_word(self, from_prefix):
        random_choice = random.random()
        i=0
        for row in self.c.execute("SELECT suffix, probability FROM markov_chain WHERE prefix=? ORDER BY RANDOM()", (from_prefix,)):
           i+=row[1] 
           if i>=random_choice:
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
            for num_words_to_try in reversed(range(1, random.randint(1, self.lookback)+1)):
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
            out = self.generate_sentence(prevent_recursion=(random.random()*10 < 1))
        
        return out
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        stderr("Usage: ./nonsense.py [[--lookback N]|[--startword Hello]|[--min-length 20]|[--max-length 300]|--no-cache] input_file_with_lotsa_words.txt")
        sys.exit(1)
    
    # default arguments
    start_word=None
    max_length=140
    no_cache=False
    lookback=3
    
    # ugly but simple enough argument handling
    args = sys.argv[1:]
    
    if "--startword" in args:
        i = args.index("--startword")
        start_word = args[i+1].decode("utf-8")
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
    
    input_sources = args
    source = args[0]
    
    stderr("Generating Markov chain for input %s…" % source)
    markov_chain = MarkovChain(input_file=args[0], lookback=lookback, no_cache=no_cache)

    stderr("Generating sentence…")
    print markov_chain.generate_sentence(start_word=start_word, max_length=max_length).encode("utf-8")
