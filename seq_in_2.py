from sklearn.preprocessing import OneHotEncoder as Encoder
import argparse
import numpy as np
import torch
import re
import os
import random
import torch.nn as nn
import torch.nn.functional as F
import math


"""

Fragmenty kodu zapożyczone z repozytorium na github'ie, z pracy M. Osipowicz
	https://github.com/marnifora/magisterka

class OHEncoder - kodowanie sewkwencji zgodnie z przyjętymi przez sieci wymaganiami;
				zawiera drobne uproszczenia względem pierwotnej wersji;

"""


class OHEncoder:

	def __init__(self, categories=np.array(['A', 'C', 'G', 'T'])):
		self.encoder = Encoder(sparse_output=False, categories=[categories])
		self.dictionary = categories
		self.encoder.fit(categories.reshape(-1, 1))
	def __call__(self,seq):
		seq = list(seq)
		info = 1
		if 'N' in seq:
			pos = [i for i, el in enumerate(seq) if el == 'N']
			if len(pos) <= 0.05*len(seq):
				info=0
				print('{} unknown position(s) in given sequence - changed to random one(s)'.format(len(pos)))
				for p in pos:
					seq[p] = random.choice(self.dictionary)
			else:
				return None
		s = np.array(seq).reshape(-1, 1)
		return self.encoder.transform(s).T, info

"""
class CustomNetwork() - inforamcje o architekturze sieci typu Custom 

"""

class CustomNetwork(torch.nn.Module):

    def __init__(self, seq_len, num_channels=[300, 200, 200], kernel_widths=[19, 11, 7], pooling_widths=[3, 4, 4],
                 num_units=[2000, 4], dropout=0.5):
        super(CustomNetwork, self).__init__()
        paddings = [int((w-1)/2) for w in kernel_widths]
        self.seq_len = seq_len
        self.dropout = dropout
        self.params = {
            'input sequence length': seq_len,
            'convolutional layers': len(num_channels),
            'fully connected': len(num_units),
            'number of channels': num_channels,
            'kernels widths': kernel_widths,
            'pooling widths': pooling_widths,
            'units in fc': num_units,
            'dropout': dropout

        }

        conv_modules = []
        num_channels = [1] + num_channels
        for num, (input_channels, output_channels, kernel, padding, pooling) in \
                enumerate(zip(num_channels[:-1], num_channels[1:], kernel_widths, paddings, pooling_widths)):
            k = 4 if num == 0 else 1
            conv_modules += [
                nn.Conv2d(input_channels, output_channels, kernel_size=(k, kernel), padding=(0, padding)),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, pooling), ceil_mode=True)
            ]
            seq_len = math.ceil(seq_len / pooling)
        self.conv_layers = nn.Sequential(*conv_modules)

        fc_modules = []
        self.fc_input = 1 * seq_len * num_channels[-1]
        num_units = [self.fc_input] + num_units
        for input_units, output_units in zip(num_units[:-1], num_units[1:]):
            fc_modules += [
                nn.Linear(in_features=input_units, out_features=output_units),
                nn.ReLU(),
                nn.Dropout(p=self.dropout)
            ]
        self.fc_layers = nn.Sequential(*fc_modules)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.fc_input)  # reshape
        x = self.fc_layers(x)
        return torch.sigmoid(x)
      
        
        
NET_TYPES = {'custom': CustomNetwork}       
"""

Część własna, parsowanie argumentów

""" 

parser = argparse.ArgumentParser(description='Program pomocniczy podający wektor 4 neuronów dla podanego modelu i zbioru sekwencji')

parser.add_argument('model',metavar='m', nargs=1, help="""Gotowy model, który chcemy 
wykorzystać do wyznaczenia wartości wektora wyników.\nPodać plik o rozszerzeniu .model\n
Plik nazwa_modelu_params.txt powinien znajdować się w tym samym katalogu""", default=None)
parser.add_argument('path',metavar='p', nargs=1, help="""Ścieżka prowadząca do katalogu z
 plikami zawierającymi testowane sekwencje""",default=os.getcwd())
parser.add_argument('ins',metavar='i', nargs=1,  help="""Niepustu plik tekstowy z nazwami plików 
fasta zawierającymi pojedyńczą sekwencję nukleotydów w przyjętym formie.""",default=None)
parser.add_argument('-o','--out', nargs=1, help="""Nazwa pliku tekstowego, do którego program 
zapisze wyniki podane przez modeli i prawdziwą kategorię sekwencji.""",default=None)


args = parser.parse_args()

if args.model is None:
	print("!!!	Podaj model\n")
elif os.path.isfile(args.model[0]):
	print(">> poprawnie podany model\n")
	modelfile = args.model[0]
	if '/' in modelfile:
		model_param=modelfile.split("/")[-1]
		name=model_param
else:
	print("!!!	Podana ścieżka nie jest poprawna lub plik nie istnieje\n")




if args.ins is None:
	print("!	Podaj plik z listą sekwencji\n")
elif os.path.isfile(args.ins[0]) or os.path.abspath(args.ins[0]):
    data_in = args.ins[0]
else:
	print("!!!	Podana ścieżka nie jest poprawna lub plik nie istnieje\n")



if not ( os.path.exists(args.path[0]) and os.path.abspath(args.path[0]) ):
	print("!!!	Podana ścieżka katalogu z sekwencjami nie istnieje.\n")
else:
	path=args.path[0]



if args.out is None:
	data_out=f'{args.ins[0].split(".")[0]}_out.txt'
	out = open(data_out,'w')
elif os.path.isfile(args.out[0]) or os.path.isfile(os.path.abspath(args.out[0])) :
	print("podany plik out istnieje\n")
	data_out = args.out[0]
	out = open(data_out,'r+')
elif not re.search("//",args.out[0]):
	data_out = args.out[0]
	out = open(data_out,'w')
else:
	print("!!!	Podana ścieżka nie jest poprawna lub plik nie istnieje\n")
	
"""
wczytywanie wybranego modelu
"""

model_param=model_param[::-1].split("_",1)[1]
model_param=model_param[::-1]
model_param=f"{modelfile.split('/',1)[0]}/{model_param}_params.txt"


par=open(os.path.normpath(model_param))
for line in par:
	if line.startswith('Network type'):
		network = NET_TYPES[line.split(':')[-1].strip().lower()]
par.close()

model = network(2000)
model.load_state_dict(torch.load(modelfile, map_location=torch.device('cpu')))
model.eval()

"""
tworzenie enkodera
"""
OHE=OHEncoder()

"""
uruchamianie modelu dla podanej listy sewkencji
"""

with open(data_in) as f:
	for sid in f:
		p=""
		if os.path.isfile(f"{path}/{sid.split()[0]}"):
			p=f"{path}/{sid.split()[0]}"
		if p:
			with open(p) as file_seq:
				for l,d in enumerate(file_seq):
					if not l:
						line= d.split()
						if re.match("^> chr[0-9XY]+ [0-9]+",d):
							name= f"{line[1][3:]}:{line[2]}"
							

						else:
							name=f"{line[1]}:{line[2]}"
							print(f"nieznany format nagłówka\nline={line}")
						out.write(f"{name}\t")
						if line[3]=='+':
							out.write(f"+\t")
						else:
							out.write(f"-\t")
						if line[4][0]=='p':
							if line[5][0]=='a':
								out.write(f"0\t")
							else:
								out.write(f"2\t")
						else:
							if line[5][0]=='a':
								out.write(f"1\t")
							else:
								out.write(f"3\t")
					else:
						seq=d.strip().upper()							
						encoded_seq=OHE(seq)
						if encoded_seq is None:
							print(name)
							out.write(f"[0.5, 0.5, 0.5, 0.5]\n")							
						else:
							X = torch.tensor(encoded_seq[0])
							X = X.reshape(1,1, *X.size())
							X=X.float()
							y=model(X)
							y=y[0].detach().numpy().tolist()
							y=str(y)
							if encoded_seq[1]:
								out.write(f"{y}\n")
							else:
								out.write(f"{y}\tzawiera_N\n")
						

out.close()
		
