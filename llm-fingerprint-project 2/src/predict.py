"""Small inference demo using the actual trained output-only CNN / LSTM model.

This is an educational classifier trained on a tiny, 3-family benchmark. Its
probabilities are model confidence scores, NOT real-world authorship evidence.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch

from src.models import TextCNN, TextLSTM
from src.train import encode


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--text',required=True,help='A response to classify')
    a.add_argument('--model',choices=['cnn','lstm'],default='cnn')
    a.add_argument('--results',type=Path,default=Path('results'))
    args=a.parse_args()
    torch.set_num_threads(2)
    cp=torch.load(args.results/f'{args.model}_output_model.pt',map_location='cpu',weights_only=True)
    vocabulary=cp['vocabulary']
    model=(TextCNN if args.model=='cnn' else TextLSTM)(len(vocabulary),len(cp['classes']),embed=cp['embed_dim'],**({'channels':cp['hidden_dim']} if args.model=='cnn' else {'hidden':cp['hidden_dim']}))
    model.load_state_dict(cp['model_state'])
    model.eval()
    ids,lengths=encode([args.text],vocabulary,256)
    with torch.no_grad():
        logits=model(ids,lengths)
        probs=torch.softmax(logits,dim=-1).squeeze().tolist()
    print('Predicted family:',cp['classes'][int(torch.tensor(probs).argmax())])
    for name,p in zip(cp['classes'],probs):
        print(f'  {name}: {p*100:.1f}%')
    print('WARNING: These are limited-benchmark classifier scores, not verified authorship probabilities.')

if __name__=='__main__':
    main()
