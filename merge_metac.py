import numpy as np
from statistics import mean

"""
Skrypt tworzący średnie poziomy metylacji/acetylacji dla podanych zbiorów sekwencji,
zarówno w wersji standaryzowanej, opisanej w pracy, jak i zwykłej średniej;
Wymaga danych dla wszystkich pacjentów z listy.

słowniki met i ac - zawierają:
	[mean , std] - średnie poziomy metylacji/acetylacji oraz odchylenie standardowe
	dla:
	[pac] 	poszczególnych pacjetów,
	[chrom]	chromosomów powyższych pacjentów, 
	
	przykładowo: met["PA01"]["1"][0] - średni poziom metylacji chromosomu 1 pacjenta PA01
	
"""

met={}
ac={}

"""
lista nazw kodowych pacjentów
"""

p=[f"DA0{x}" for x in [1,3,4,5,6]]+[f"GB0{x}" for x in range(1,9)]+["PA01","PA03","PA04"]
p_met=p+["PA05","PA06","PA07"]
p_ac=p+["PA02"]
chrom=[str(x) for x in range(1,23)]+["X","Y"]

"""
ds - 	numer zbioru danych
lines - liczba sekwencji w zbiorze danych


dla ds3 - 10000
dla ds6 - 40000
"""
ds=3
lines=10000

for pac in p_met:
	with open(f"/media/mateusz/dysk_E/licencjat/poziomy/{pac}_met.txt","r") as f:
		met[pac]={}
		for l,line in enumerate(f):
			line=line.strip().split("\t")
			met[pac][chrom[l]]=(float(line[0]),float(line[1]))

for pac in p_ac:
	with open(f"/media/mateusz/dysk_E/licencjat/poziomy/{pac}_ac.txt","r") as f:
		ac[pac]={}
		for l,line in enumerate(f):
			line=line.strip().split("\t")
			ac[pac][chrom[l]]=(float(line[0]),float(line[1]))


ki=["pa","na","pi","ni"]

M= np.zeros((4,len(p_met),lines))
M1=np.zeros((4,len(p_met),lines))

"""
N - macierz zawirająca numer chromosomu i pozycję środka sekwencji ( razem tworzą standardowy id, tzn.: 1:21245 )
"""

N= np.zeros((4,lines,2),dtype=np.dtype('U16'))

for k in range(4):
	with open(f"/media/mateusz/dysk_E/licencjat/poziomy/ds{ds}_id_{ki[k]}.bed") as f:
		for l,line in enumerate(f):
			line=line.strip().split("\t")
			N[k][l][0]=line[0][3:]
			N[k][l][1]=str(int(line[1])+1000)

"""
przerabianie metylacji
"""
			 
for k in range(4):
	for pp,pac in enumerate(p_met):
		with open(f"/media/mateusz/dysk_E/licencjat/poziomy/{pac}_ds{ds}_{ki[k]}_met.txt","r") as f:
			for l,line in enumerate(f):
				line=line.strip()
				if line == "None":
					line=met[pac][N[k][l][0]][0]
				else:
					line=float(line)
				M[k][pp][l]=(line-met[pac][N[k][l][0]][0])/met[pac][N[k][l][0]][1]					# standaryzajca (x-m)/SE dla odpowiedniego chromosomu i pacjenta
				M1[k][pp][l]=line																	# wersja niestandaryzowana
	outn=open(f"/media/mateusz/dysk_E/licencjat/poziomy/ds{ds}_{ki[k]}_met_norm.txt","w+")
	out=open(f"/media/mateusz/dysk_E/licencjat/poziomy/ds{ds}_{ki[k]}_met.txt","w+")
	for l in range(lines):
		outn.write(f"{N[k][l][0]}:{N[k][l][1]}	{mean(M[k,:,l])}\t{k}\n")
		out.write(f"{N[k][l][0]}:{N[k][l][1]}	{mean(M1[k,:,l])}\t{k}\n")
	outn.close()
	out.close()

"""
przerabianie acetylacji
"""
M= np.zeros((4,len(p_met),lines))
M1=np.zeros((4,len(p_met),lines))

for k in range(4):
	for pp,pac in enumerate(p_ac):
		with open(f"/media/mateusz/dysk_E/licencjat/poziomy/{pac}_ds{ds}_{ki[k]}_ac.txt","r") as f:
			for l,line in enumerate(f):
				line=line.strip()
				if line == "None":
					line=ac[pac][N[k][l][0]][0]
				else:
					line=float(line)
				M[k][pp][l]=(line-ac[pac][N[k][l][0]][0])/ac[pac][N[k][l][0]][1]					# standaryzajca (x-m)/SE dla odpowiedniego chromosomu i pacjenta
				M1[k][pp][l]=line																	# wersja niestandaryzowana
	outn=open(f"/media/mateusz/dysk_E/licencjat/poziomy/ds{ds}_{ki[k]}_ac_norm.txt","w+")
	out=open(f"/media/mateusz/dysk_E/licencjat/poziomy/ds{ds}_{ki[k]}_ac.txt","w+")
	for l in range(lines):
		outn.write(f"{N[k][l][0]}:{N[k][l][1]}	{mean(M[k,:,l])}\t{k}\n")
		out.write(f"{N[k][l][0]}:{N[k][l][1]}	{mean(M1[k,:,l])}\t{k}\n")
	outn.close()
	out.close()

