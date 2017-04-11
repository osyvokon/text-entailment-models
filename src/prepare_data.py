#!/usr/bin/env python3
"""Convert SNLI dataset to the simplified format.
"""
import argparse
import json
import os
import re



def main(snli_path, output_path):
    """Convert SNLI dataset file by file.

    Args:
        snli_path (str): path to the extracted snli_1.0 dir.
        output_path (str): path to the output directory.

    """

    assert os.path.isdir(snli_path)
    outputs = {
        'snli_1.0_dev.jsonl': 'dev.txt',
        'snli_1.0_train.jsonl': 'train.txt',
        'snli_1.0_test.jsonl': 'test.txt',
    }
    print("Converting data files")
    for src, dest in outputs.items():
        dest_path = os.path.join(output_path, dest)
        with open(dest_path, 'w') as f:
            print('   {}'.format(dest_path))
            converted = convert_file(os.path.join(snli_path, src))
            f.write('\n'.join(converted))


def convert_file(path):
    """Convert a single file in .jsonl format to the simplified format. """

    with open(path) as f:
        for line in f:
            yield convert_line(line)


def convert_line(line):
    """Convert a single .jsonl line to the simplified format. """

    line = json.loads(line)
    s1 = ' '.join(tokenize(line['sentence1']))
    s2 = ' '.join(tokenize(line['sentence2']))
    label = line['gold_label']

    return '\t'.join([label, s1, s2])


def tokenize(s):
    """Simple word tokenizer.

    Example:
        >>> tokenize('Hello, world!')
        ['hello', 'world']

    """

    return [w.lower() for w in re.findall(r'\w+', s)]


def test_convert_line():
    line = """{"annotator_labels": ["neutral", "entailment", "neutral", "neutral", "neutral"], "captionID": "4705552913.jpg#2", "gold_label": "neutral", "pairID": "4705552913.jpg#2r1n", "sentence1": "Two women are embracing while holding to go packages.", "sentence1_binary_parse": "( ( Two women ) ( ( are ( embracing ( while ( holding ( to ( go packages ) ) ) ) ) ) . ) )", "sentence1_parse": "(ROOT (S (NP (CD Two) (NNS women)) (VP (VBP are) (VP (VBG embracing) (SBAR (IN while) (S (NP (VBG holding)) (VP (TO to) (VP (VB go) (NP (NNS packages)))))))) (. .)))", "sentence2": "The sisters are hugging goodbye while holding to go packages after just eating lunch.", "sentence2_binary_parse": "( ( The sisters ) ( ( are ( ( hugging goodbye ) ( while ( holding ( to ( ( go packages ) ( after ( just ( eating lunch ) ) ) ) ) ) ) ) ) . ) )", "sentence2_parse": "(ROOT (S (NP (DT The) (NNS sisters)) (VP (VBP are) (VP (VBG hugging) (NP (UH goodbye)) (PP (IN while) (S (VP (VBG holding) (S (VP (TO to) (VP (VB go) (NP (NNS packages)) (PP (IN after) (S (ADVP (RB just)) (VP (VBG eating) (NP (NN lunch))))))))))))) (. .)))"}"""
    expected = (
        'neutral\t'
        'two women are embracing while holding to go packages\t'
        'the sisters are hugging goodbye while holding to go packages after just eating lunch')
    assert convert_line(line) == expected


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('snli_path', help='Path to the snli_1.0 dir')
    parser.add_argument('output_path', help='Output directory')
    args = parser.parse_args()
    main(args.snli_path, args.output_path)
