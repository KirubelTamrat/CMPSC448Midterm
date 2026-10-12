"""RQ4 control: can simple, interpretable response-style features identify model families?

Uses the exact same prompt-disjoint split as the neural experiments. Trains only
on the training portion (ignoring validation), evaluates on untouched test set.
This is an additional diagnostic baseline, not a replacement for CNN or LSTM.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.train import FAMILIES, feature_row, load_data, metrics, partition


def x_values(frame, columns):
    return np.array([[feature_row(s)[c] for c in columns]
                     for s in frame.LLM_output], dtype=float)


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--data',type=Path,default=Path('data/llm_responses.csv'))
    a.add_argument('--out',type=Path,default=Path('results/style_baseline.json'))
    a.add_argument('--seed',type=int,default=42)
    args=a.parse_args()
    d=load_data(args.data)
    train, val, test=partition(d,args.seed)
    y_train=np.array([FAMILIES.index(x) for x in train.LLM_name])
    y_test=np.array([FAMILIES.index(x) for x in test.LLM_name])
    configs={'word_count_only':['response_words'],
             'seven_style_features':['response_words','avg_word_length','lexical_diversity','average_sentence_words','markdown_symbols','bullet_lines','newlines']}
    result={'train_responses':len(train),'test_responses':len(test),'test_prompts':test.prompt_group.nunique(),'models':{}}
    for name,columns in configs.items():
        clf=make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000,random_state=args.seed))
        clf.fit(x_values(train,columns),y_train)
        ypred=clf.predict(x_values(test,columns))
        result['models'][name]={'columns':columns,**metrics(y_test,ypred)}
        print(f"{name}: accuracy {result['models'][name]['accuracy']:.3f}; macro F1 {result['models'][name]['macro_f1']:.3f}")
    args.out.parent.mkdir(exist_ok=True,parents=True)
    args.out.write_text(json.dumps(result,indent=2))
    print('Saved:',args.out)

if __name__=='__main__':
    main()
