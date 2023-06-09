import argparse
import numpy as np
import os
import re

"""
Skrypt łączący wyniki z podanych modeli do jednego pliku tekstowego.
Fromat danych wyjściowych:
id;	nić;	prawdziwa kat.; custom40 kat.; custom41 kat.; alt1 kat.; alt2 kat.; custom40 wektor; custom41 wektor; alt1 wektor; alt2 wektor;

Zakłada się, że wszystkie 4 pliki wejściowe są podane w określonej kolejności 
oraz sawierają sekwecnjie należące do tej samej kategorii.

parsowanie argumnetów
"""

nazwy=["custom40", "custom41", "alt1", "alt2"]

parser = argparse.ArgumentParser(description='Program pomocniczy łączący wyniki z modeli dla tego samego zbioru danych wejściowych',formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-p','--path', nargs=1, help="""Ścieżka prowadząca do katalogu z plikami 
zawierającymi pliki do złączenia""",default=os.getcwd())
parser.add_argument('-i','--ins', nargs=len(nazwy), help=f"""Lista nazw plików, które chcemy złączyć;
\nPowinna zawierać wyniki z modeli w kolejnosci:\ncustom40\tcustom41
alt1\t\talt2\n>>> Nazwy plików powinny zawierać wzorce: custom lub alt z odpowieczną liczbą""",default=None)
parser.add_argument('-o','--out', nargs=1, help="""Nazwa pliku tekstowego,do którego zapiszemy 
złączone dane\n\nFormat danych wyjściowych:\n
id;\tnić;\tprawdziwa kat.;\tcustom40 kat.;\tcustom41 kat.;\talt1 kat.;\talt2 kat.;\tcustom40 
wektor;\tcustom41 wektor;\talt1 wektor;\talt2 wektor;""",default=None)
parser.add_argument('-l','--lines', nargs=1, help="Oczekiwana liczba wierszy do złączenia; domyślnie 10000",default=10000)


args = parser.parse_args()



file_name=["" for i in range(len(nazwy))]

"""
funkcja which(line) określa na podstawie wektora wynikowego każdej sekwencji, 
	do której kategorii został przypisany; zawsze wybiera kategorię o największej
	wartości neuronu nyjściowego;
line - string, wektor wynikowy z danej sieci, np: "[0.5, 0.9894, 0.7521, 0.5]"
"""
def which(line):
	line=line[3][1:]
	line=line[:-2]
	rob=[float(l) for l in line.split(",")]
	wynik=0
	pom=0
	for y in range(4):
		if pom<rob[y]:
			wynik=y
			pom=rob[y]
	return wynik



if args.out is None or args.ins is None:
	print("\nPodaj pliki wejściowe i/lub nazwę pliku wyjściowego.\n")
else:
	print("Sprawdzenie poprawności nazw plików wejściowych:\n")
	ile=[0,0,0,0]
	path=args.path[0]
	if isinstance(args.lines, int):
		
		lines=args.lines
	else:
		lines=int(args.lines[0])
	print(lines)
	for ind,n in enumerate(nazwy):
		for w in args.ins:
			if re.search(n,w):
				print(f"> Dane z modelu {n} znalezione\n")
				ile[ind]=ile[ind]+1
				if ile[ind]>1:
					ile[ind]=ile[ind]+1
				file_name[ind]=w
	if sum(ile)!=len(nazwy):
		print("Nieporawnie wprowadzone dane lub dane z nazwami o niepoprawym formacie")
	else:
		f_c40=open(f"{path}/{file_name[0]}")
		f_c41=open(f"{path}/{file_name[1]}")
		f_a1=open(f"{path}/{file_name[2]}")
		f_a2=open(f"{path}/{file_name[3]}")
		
		f_out=open(f"{path}/{args.out[0]}","w")
		f_list=[f_c40,f_c41,f_a1,f_a2]
		dane=["" for x in range(len(nazwy))]
		for ind in range(lines):
			c40=f_c40.readline().split("\t")
			dane[0]=c40[3][:-1]
			f_out.write(f"{c40[0]}\t{c40[1]}\t{c40[2]}\t{which(c40)}\t")
			for x in range(1,len(nazwy)):
				d=f_list[x].readline().split("\t")
				f_out.write(f"{which(d)}\t")
				dane[x]=d[3][:-1]
			for x in range(len(nazwy)-1):
				f_out.write(f"{dane[x]}\t")
			f_out.write(f"{dane[len(nazwy)-1]}\n")	



		f_c40.close()
		f_c41.close()
		f_a1.close()
		f_a2.close()			
		f_out.close()

