import argparse
import os

"""
Skrypt konwertujący przyjęte wewnątrz pracy id sekwencji, tzn.: nr_chrom:środekowa pozycja
do formatu bed, tzn.: chr{nr_chrom}	{start}	{stop}
"""

parser = argparse.ArgumentParser(description=f'Konwersja id sekwencji na pliki bed',formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('i', nargs=1, help="Ścieżka do pliku z id sekwencji w formacie nr_chr:środkowa_pozycja, np. 10:254000",default=None)
parser.add_argument('--out', nargs=1, help="Ścieżka i nazwa pliku wyjściowego, domyślnie {--in}.bed",default=None)

args = parser.parse_args()

if args.out is None:
	out=str(args.i[0]).split(".")[0]+".bed"
else:
	out=args.out[0]

o= open(out,"w+")

with open(args.i[0]) as f:
	for line in f:
		line=line.strip().split(":")
		o.write(f"chr{line[0]}\t{max([int(line[1])-1000,0])}\t{int(line[1])+1000}\n")
o.close()
