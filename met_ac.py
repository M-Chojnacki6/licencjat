import pyBigWig

"""
skrypt znajdujący średnie poziomy metylacji dla podanych na 
ścieżce n2 obszarów chromodomu dla poszczególnych pacjentów.

zmienne:
> nazwy 	- zawiera kodowe nazwy pacjentów
> czy_ac 	- True - wypisuje średni poziom acetylacji
			- Flase - średni poziom metylacji
> dataset	- numer interesującego nas zbioru danych.
"""

#nazwy= ["PA03","PA04","PA05","PA05","PA06","PA07"]

czy_ac=True
dataset="6"

if czy_ac:
	nazwy=["DA01","DA03","DA04","DA05","DA06","GB01","GB02","GB03","GB04","GB05","GB06","GB07","GB08","PA01","PA02","PA03","PA04"]
	his="H3K27ac"
	h="ac"
	
else:
	nazwy=["DA01","DA03","DA04","DA05","DA06","GB01","GB02","GB03","GB04","GB05","GB06","GB07","GB08","PA01","PA03","PA04","PA05","PA05","PA06"]
	his="H3K4me3"
	h="met"
	
for nazwa in nazwy:
	n0="/home/mchojnacki/licencjat/"+his+"/"+nazwa+"_coverage_"+his+".bigwig"
	bw = pyBigWig.open(n0)
	for n in ["pa","pi","na","ni"]:
		n1=nazwa+"_ds"+dataset+"_"+n+"_"+h+".txt"
		
		n2="/home/mchojnacki/licencjat/poziomy/ds"+dataset+"_id_"+n+".bed"
		
		out=open(n1,"w")
		with open(n2,"r") as f:
			for line in f:
				line=line.strip().split("\t")
				M = bw.stats(line[0],int(line[1]),int(line[2]))
				out.write(str(M[0]))
				out.write("\n")
		out.close()
	bw.close()
