import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import binomtest, fisher_exact, chi2_contingency, mannwhitneyu,chisquare



nazwy=["custom40", "custom41", "alt1", "alt2"]
ni=["c40","c41","a1","a2"]

parser = argparse.ArgumentParser(description=f'Wstępna analiza danych z {len(nazwy)} sieci: {nazwy}\nwszystkie pliki wyjściowe od flag -s zapisywane są w katalogu \zbiory w bieżącym katalogu',formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('path',metavar="p", nargs=1, 
help="""Ścieżka prowadząca do katalogu z plikami zawierającymi złączone piki z 4 kategorii;""",default=os.getcwd())
parser.add_argument('-o','--out', nargs=1, 
help="""Ścieżka prowadząca do katalogu, w którym skrypt ma zapisać wykresy; w przypadku braku tylko wyświetla wykresy;""",default=None)
parser.add_argument('-l','--lines', nargs=1, 
help="""Oczekiwana liczba wierszy w plikach wejściowych; domyślnie 10000;""",default=10000)
parser.add_argument('-w1', 
help="""tworzy wykresy liczby poprawnie i błędnie przypisanych kategorii z podziałem na sieci i kategorie;""",action='store_true', default=False)
parser.add_argument('-w2', help="""tworzy dla każdej sieci wykres przyporządkowania sekwencji do kategorii z podziałem na chromosomy;""",
action='store_true', default=False)
parser.add_argument('-w3', nargs="+",help="""dla każdej sekwencji z porównywanej pary tworzy wykres pokazujący podobieństwa błędów pomiedzy sieciami
wprowadzanie pary porównywanych sieci: sieć1_sieć2\n0 - custom40\n1 - custom41\n2 - alt1\n3 - alt2;""", default=False)
parser.add_argument('-w4', nargs="+",help="""dla danych z podanych sieci tworzy:
1) wykres słupkowy liczby błędnego przyporządkownaia podanej pary sieci do poszczególnych kategorii
2) mapę ciepła zliczającą podwójne błędy dla danej sekwencji
wprowadzanie pary porównywanych sieci: sieć1_sieć2, np. 0_1\n0 - custom40\n1 - custom41\n2 - alt1\n3 - alt2""", default=False)
parser.add_argument('-w5', nargs="+", type=int,
help="""tworzy wykresy słupkowe jednoznacznych i niejednoznacznych przyporządkowań (1 vs. wiele wzbudzonych) 
z podziałem na liczbę błędów w przyporządkowaniu;""", choices=[0,1,2,3,4],default=False)
parser.add_argument('-s1', nargs="+", type=int,
help="""wypisuje do plików identyfikatory sekwencji, przyporządkownia i wyniki sieci dla sieci typu S1:
1) błędne przyporządkowanie w tę samą stronę obu sekwencji [ 1 plik zmiana_p ]
2) błędne przyporządkowanie w przynajmniej przez 1 sieć [4 pliki zmiana_{prawdziwa}];
0 - sieci custom\n1 - sieci alt""",choices=[0,1], default=False)
parser.add_argument('-s2', nargs="+", type=int,
help="""wypisyje do pliku sekwencjie, które pomyliły się w tę samą stronę  dla >=S2 sieci
zwraca 1 plik zmana_{s2}.txt w formacie:

id;	prawdziwa_kategoria; kategoria_większości;	kat_custom40; kat_custom41; kat_alt1; kat_alt2""",choices=[0,1,2,3,4], default=False)
parser.add_argument('-s3', nargs="+", type=int,
help="""wypisuje do plików identyfikatory sekwencji, przyporządkownia i wyniki sieci dla sekwencji 
przyporządkowanych poprawnie >=s3 razy
do pliku stan_p.txt
format:
id;	prawdziwa_kategoria; 	błędnia_kategoria; 	kat_sieci x 4; rodzaj_pewności_wektora x 4""",choices=[0,1,2,3,4], default=False)
parser.add_argument('-s4', nargs="+", type=int,
help="""wypisuje do plików identyfikatory sekwencji, przyporządkownia i wyniki sieci dla sekwencji 
przyporządkowanych niejednoznacznie >=ile razy
do pliku niepewny_{ile}.txt
format:
id;	prawdziwa_kategoria; 	błędnia_kategoria; 	kat_sieci x 4; rodzaj_pewności_wektora x 4""",choices=[0,1,2,3,4], default=False)
parser.add_argument('-s5', nargs="+", type=int,
help="""wypisuje do plików identyfikatory sekwencji i prawdziwe kategorie sieci dla sekwencji 
przyporządkowanych jednoznacznie >=ile razy do plików:

jendozn_{ile}_{kat}_{liczba błędnych przyporządkowań}.txt
niejednozn_{ile}_{kat}_{liczba błędnych przyporządkowań}.txt 
w formacie 
id;	prawdziwa kategoria
przyjmuje dane o ile""", choices=[0,1,2,3,4],default=False)
parser.add_argument('-stat1', 
help="""test istotności z rozkładu dwumianowego,
wypisuje p-wartości dla 4 sieci obrazujące istotność hipotezy
h0 - uzyskane_błędy = oczekiwane_błędy
h1 - uzyskane_błędy != oczekiwane_błędy
h2 - uzyskane_błędy > oczekiwane_błędy\n""",action='store_true', default=False)

parser.add_argument('-stat2', 
help="""dokładny test Fisher'a do sprawdzenia, czy dwie sieci popełniają błędny 
w analogicznych miejscach dla wszystkich par sieci, hipotezy:
h0 - zmienne niezależne
h1 - zmienne zależne\n""",action='store_true', default=False)
parser.add_argument('-stat3', 
help="""test zgodności Chi^2 Pearsona do do sprawdzenia, czy dwie sieci popełniają błędny 
w analogicznych miejscach dla wszystkich par sieci, hipotezy:
h0 - zmienne niezależne
h1 - zmienne zależne\n""",action='store_true', default=False)
parser.add_argument('-stat4', type=int,
help="""test U Manna-Whitney'a do porówania średnich ilości popełnionych
błędów przy przyporządkowywaniu sekwecji, sekwencje przyporzadkowane jednoznacznie przez >=stat4 sieci kontra reszta""",choices=[1,2,3], default=False)




args = parser.parse_args()

global typ
typ=0

if args.out is None:
	out=None
else:
	out=args.out[0]
	out=f"{out}/"
	out=out.replace("//","/")
	if not os.path.exists(out):
		out=f"{os.getcwd()}/{out}"
		out=out.replace("//","/")
		if not os.path.exists(out):
			print(f"Podany katalog: {out} \nprzeznaczony do zapisu plików nie istnieje")
			out=None

path=args.path[0]

if not os.path.exists(path):
	print("Podany katalog: {path}\nz danymi z sieci nie istnieje")

global skala_w
global skala_n
global ff

if "3" in path:
	skala_w=[0,2000,4000,6000,8000,10000]
	skala_n=["0", "2k", "4k", "6k", "8k", "10k"]
	typ=11
	ff="10k"
elif "6" in path:
	skala_w=[0,8000,16000,24000,32000,40000]
	skala_n=["0", "8k", "16k", "24k", "32k", "40k"]
	typ=7
	ff="40k"

if isinstance(args.lines, int):
	lines=args.lines
else:
	lines=int(args.lines[0])
	
"""
F_mod - macierz zwierająca rozkłady oczekiwanej poprawności przyporządkawania 
sekwencji do poprwnej kategorii z podziałem na kategorie
		wymiary: 4 x 4 (model, kategoria)

"""
global F_mod

F_mod =np.array([[0.913,0.818,0.805,0.852],[0.925,0.835,0.808,0.854],[0.914,0.790,0.733,0.830],[0.912,0.796,0.735,0.831]])

"""
funkcja wczytaj_sl(ac, met) wczytuje do słowników dane o acetylacji i metylacji danych sekwecnji:

format słownika:
	id_seq : (poziom_[met/ac], prawdziwa_kategoria)
"""	
def wczytaj_sl(ac,met):
	global dict_met
	global dict_ac
	dict_met={}
	dict_ac={}
	with open(ac,"r") as pac:
		for line in pac:
			line=line.strip().split("\t")
			dict_ac[line[0]]=(line[1],line[2])
	with open(met,"r") as pmet:
		for line in pmet:
			line=line.strip().split("\t")
			dict_met[line[0]]=(line[1],line[2])

"""
m1 - macierz 4 x len(nazyw) sumująca błednie przypasowanae sekwencje w obrębie kategorii
	prawdziwa_kat x dana_sieć
	
m2 - macierz len(nazyw) x 4 x len(nazyw) pokazująca kategorie przypisane przez modele
	nr_sekwencji x prawdziwa_kat x dana_sieć
	
m3 - macierz len(nazyw) x 4 zawierająca identyfikatory i prawdziwe kategorie
	nr_sekwencji x prawdziwa_kat
	
m4 - macierz len(nazyw) x 4 x len(nazyw) x 4 zawirająca wektory wynikowe z sieci
	nr_sekwencji x prawdziwa_kat x dana_sieć

"""
global m1
global m2
global m3
global m4
m1=np.zeros((4,len(nazwy)))
m2=np.empty((lines,4,len(nazwy)),dtype=np.int8)
m3=np.empty((lines,4),dtype=np.dtype('U16'))
m4=np.empty((lines,4,len(nazwy),4))

"""
funkcja fun() wczytuje linie z plików wejściowych do podanych wyżej struktur reprezentacji danych
"""
def fun(line,kat,nr):
	for x in range(len(nazwy)):
		m2[nr][kat][x]=line[3+x]
		m3[nr][kat]=line[0]
		rob=line[typ+x]
		rob=rob.strip()
		rob=rob[1:-1]
		rob=rob.split(",")
		for ri, r in enumerate(rob):
			m4[nr][kat][x][ri]=round(float(r),4)
		
		
		if line[2]!=line[3+x]:
			m1[kat][x]=m1[kat][x]+1

"""
funkca w1() tworzy wykresy liczby poprawnie i błędnie przypisanych kategorii z podziałem na sieci i kategorie
patrz Rysunek 3.1

"""

def w1():
	fig, ax = plt.subplots(2,2,figsize=(int(len(nazwy)/2)*5,10))
	axs=[ax[0,0],ax[1,0],ax[0,1],ax[1,1]]
	bottom = np.zeros(4)
	width=0.5
	fig.suptitle("Skategoryzowne poprawnie vs. błędnie",fontsize=16)
	for lab,dane,a in zip(nazwy,m1.T,axs):
		a.set_ylim(0,lines)
		v=a.bar(ki, dane, width=0.9, label="błędnie", bottom=bottom)
		a.bar_label(v, label_type='center',fontsize=16)
		w=[lines for x in range(4)]-dane
		v=a.bar(ki, w, width=0.90, label="poprawnie", bottom=dane)
		a.set_xticks(ticks=[0,1,2,3],labels=ki,fontsize=14)
		a.set_yticks(ticks=skala_w,labels=skala_n,fontsize=16)
		a.set_title(lab,fontsize=20)
		a.bar_label(v, label_type='center',fontsize=16)
	a.legend(fontsize=14,bbox_to_anchor=(1.1,2.5))  #loc="upper left",
	if out:
		fig.savefig(f"{out}{ff}_w1.png")
		plt.close()
	else:
		plt.show()


"""
funkcja w2() tworzy dla każdej sieci wykres przyporządkowania sekwencji do kategorii z podziałem
na chromosomy;
patrz Rysunek 3.2
"""


def w2():
	chrom=[str(x) for x in range(1,23)]+["X","Y"]
	start=np.zeros((22,4),dtype=int)
	mm=np.zeros((lines,4),dtype=int)
	for kat in range(4):
		xx=0
		for ci,c in enumerate(chrom):
			for l,line in enumerate(m3[:,kat]):
				if line.split(":")[0]==c:
					mm[xx,kat]=l
					xx+=1
			if ci<22:
				start[ci,kat]=xx
	chrom=chrom[:22]
	chrom.append("X/Y")
	xl=np.zeros((23,4))	
	xs=np.zeros(lines)
	ys=np.ones(lines)			
	for mod in range(len(nazwy)):
		fig, ax = plt.subplots(4,figsize=(15,10))
		fig.suptitle(f"przyporządkowania w sieci {nazwy[mod]}",fontsize=24)
		fig.supxlabel("chromosom",fontsize=20)
		fig.supylabel("kategoria",fontsize=20)


		for kat,a in enumerate(ax):
			for x in range(lines):
				xs[x]=m2[mm[x,kat],kat,mod]
			xl[0,kat]=start[0,kat]/2
			for x in range(21):
				xl[x+1,kat]=(start[x,kat]+start[x+1,kat])/2
			xl[22,kat]=(start[21,kat]+lines)/2
			a.set_xlim(0,lines)
			a.set_ylim(-0.25,3.25)
#			a.set_title(f"kategoria: {ki[kat]}")
			a.vlines(start[:,kat],ymin=-0.25,ymax=3.25,colors="b",linestyles='dotted',label ="granice chromosomów")
			a.plot(xs,"r.",ms=0.25,label = "błędy")
			a.plot(ys*kat,"g-",linewidth=2.5,label ="poprawna kat.")
			a.set_yticks(np.arange(4),ki,fontsize=14)
			a.set_xticks(ticks=xl[:,kat],labels=chrom,fontsize=12)
			if not kat:
				a.legend(fontsize=14,bbox_to_anchor=(0.75,1.08),markerscale=10)
		if out:
			fig.savefig(f"{out}{ff}_w2_{ni[mod]}.png")
			plt.close()
		else:
			plt.show()


"""
funkcja w3(mod1, mod2) dla każdej sekwencji tworzy wykres pokazujący podobieństwa błędów pomiedzy
modelami

* mod1 i mod2 to int numeru porządkowego typu sieci (patrz lista nazwy)

patrz Rysunek 3.3

"""	
	
def w3(mod1,mod2):
	lab=["obie poprawnie", f"błąd {nazwy[mod1]}",f"błąd {nazwy[mod2]}", "obie źle"]
	
	dane,suma=p1(mod1,mod2,3)
	
	fig2,ax2= plt.subplots(figsize=(10,10))
	fig2.supxlabel("kategoria",fontsize=20)
	fig2.supylabel("liczebność",fontsize=20)
	ax2.set_xticks(ticks=[0,1,2,3],labels=ki,fontsize=16)
	ax2.set_yticks(ticks=skala_w,labels=skala_n,fontsize=16)
	fig2.suptitle(f"Porównanie przyporządkowań sieci {nazwy[mod1]} vs. {nazwy[mod2]}",fontsize=24)
	bottom = np.zeros(4)
	for x,s in enumerate(suma.T):
		ax2.set_ylim(0,lines)
		a=ax2.bar(ki, s,width=0.9, label=lab[x], bottom=bottom)
		bottom=bottom+s
		ax2.bar_label(a, label_type='center',fontsize=20)


	fig2.legend(bbox_to_anchor=(0.37,0.25),fontsize=14)	# "upper right"
	if out:
		fig2.savefig(f"{out}{ff}_w3_{ni[mod1]}_{ni[mod2]}_2.png")
		plt.close()
	else:
		plt.show()
	
"""
funkcja p1(mod1,mod2,wyk) dla podanych modeli zwraca:
	* dane 		- macierz 1 x lines zawierającą:
		-> 0 dla podwójnego poprawnego przyporządkowania do kategorii z mod1 i mod2
		-> 1 dla popawnego z mod2 i błednego z mod1
		-> 2 dla popawnego z mod1 i błednego z mod2
		-> 3 dla obu przyporządkowanych błędnie
	* suma 		- macierz 4 x 4, zawierającą w wierszach zliczenia wymienionych klas na odpowiednich pozycjach
	* kierunek 	- macierz 4 x 4 x 4 zliczeń w której poszczególne współrzedne oznaczają:
		-> 1. - ["p.a.","np.a.","p.in.","np.in."]
		-> 2. - [błędną wartość mod1, błędną wartość mod1, obie błędne takie same war mod1,obie błędne takie same war mod2]
		-> 3. - [0, 1, 2, 3] - błedną warość
	* w_kier - macierz 4 x 4 X 4  zliczająca kombinacje podwójnych błędów: 
		-> 1. - ["p.a.","np.a.","p.in.","np.in."]
		-> 2. - [0, 1, 2, 3] - błedną warość mod1
		-> 3. - [0, 1, 2, 3] - błedną warość mod2
	* Fisher - zwraca macierz 4 x 2 x 2 odpowiednik 4 macierzy zliczeń Fishera w postaci:
		kat x
			[[(błąd_1 i błąd_2), 		(błąd_1 i poprawny_2)],
			 [(poprawny_1 i błąd_2), 	(poprawny_1 i poprawny_2)]]
	* wyk - int, liczba z nazyw funkcji, która tworzy wykres:
		1, 3, 4, ...
"""

def p1(mod1,mod2,wyk):
	dane=np.zeros(lines)
	suma=np.zeros((4,4))
	Fisher=np.zeros((4,2,2),dtype=int)
	kierunek=np.zeros((4,4,4))
	w_kier=np.zeros((4,4,4),dtype='h')
	for kat in range(4):
		for l in range(lines):
			if kat==m2[l][kat][mod1]:
				if kat==m2[l][kat][mod2]:
					dane[l]=0
					suma[kat][0]+=1
					Fisher[kat][1][1]+=1
				else:
					dane[l]=2
					suma[kat][2]+=1
					kierunek[kat][1][m2[l][kat][mod2]]+=1
					Fisher[kat][1][0]+=1
			else:
				if kat==m2[l][kat][mod2]:
					dane[l]=1
					suma[kat][1]+=1
					kierunek[kat][0][m2[l][kat][mod1]]+=1
					Fisher[kat][0][1]+=1
				else:
					dane[l]=3
					suma[kat][3]+=1
					kierunek[kat][2][m2[l][kat][mod1]]+=1
					kierunek[kat][3][m2[l][kat][mod2]]+=1
					w_kier[kat][m2[l][kat][mod1]][m2[l][kat][mod2]]+=1
					Fisher[kat][0][0]+=1
	if wyk==3:
		return dane, suma
	elif wyk==4:
	
		return kierunek, w_kier
	elif wyk==1:
		return Fisher
	
"""
funkcja w4(mod1,mod2) dla danych z podanych modeli tworzy:
1) wykres słupkowy liczby błędnego przyporządkownaia podanej pary modeli do poszczególnych kategorii
2) mapę ciepła zliczającą podwójne błędy dla danej sekwencji
patrz Rysunek 3.4
"""

def w4(mod1,mod2):
	kier, wkier = p1(mod1, mod2, 4)
	
	lab=[ f"błąd {nazwy[mod1]}",f"błąd {nazwy[mod2]}", f"oba źle\nbłąd {nazwy[mod1]}",f"oba źle\nbłąd {nazwy[mod2]}"]
	fig, ax = plt.subplots(2,2,figsize=(20,20))
	fig.suptitle(f"Zależności liczby błędów\nmodel {nazwy[mod1]} vs. {nazwy[mod2]}")
	fig.supylabel("liczebność")
	fig.supxlabel("rodzaj błędu")
	axs=[ax[0,0],ax[0,1],ax[1,0],ax[1,1]]
	
	fig2,ax2= plt.subplots(2,2,figsize=(20,20))
	axs2=[ax2[0,0],ax2[0,1],ax2[1,0],ax2[1,1]]
	fig2.suptitle(f"Rodzaj błędu podwójnego\nsieć {nazwy[mod1]} vs. {nazwy[mod2]}",fontsize=20)
	fig2.supylabel(f"sieć {nazwy[mod1]}",fontsize=16)
	fig2.supxlabel(f"sieć {nazwy[mod2]}",fontsize=16)
	for kat in range(4):
		axs[kat].set_title(f"kategoria: {ki[kat]}",fontsize=16)
		axs2[kat].set_title(f"kategoria: {ki[kat]}",fontsize=20)
		bottom=[0,0,0,0]
		for r,d in enumerate(kier[kat].T):
			a=axs[kat].bar(lab, d,0.5, bottom=bottom, label=f"{ki[r]}")
			bottom=bottom+d
			if sum(d):
				axs[kat].bar_label(a, label_type='center',fontsize=20)
		if not kat:
			fig.legend(title="przypisana kategoria")
		
		axs2[kat].set_yticks(np.arange(4),labels=ki,fontsize=18)
		axs2[kat].set_xticks(np.arange(4),labels=ki,fontsize=18)
		im=axs2[kat].imshow(wkier[kat])
		cbar=plt.colorbar(im,ax=axs2[kat])
		for i in range(4):
			for j in range(4):
				text = axs2[kat].text(j,i,wkier[kat][i,j],ha="center",ma="center",color="w",fontsize=16)
	if out:
		fig.savefig(f"{out}{ff}_w4_{ni[mod1]}_{ni[mod2]}.png")
		fig2.savefig(f"{out}{ff}_w4_{ni[mod1]}_{ni[mod2]}_2.png")
		plt.close()
	else:
		plt.show()
	

"""
funkja kategory(wektor) klasyfikuje sekwencję do jednej z 16 kategorii:

oznaczenia: 1 - neuron wzbudzony,		tzn. >0.5
			0 - neuron niewzbudzony,	tzn. =0.5 

* 0,1,2,3 oznaczają jednoznaczne przyporządkowanie do danego neuronu, tzn: jedna wartość !=0.5 reszta 0.5
	0:	[1, 0, 0, 0] promoter active 
	1:	[0, 1, 0, 0] nonpromoter active
	2:	[0, 0, 1, 0] promoter inactive
	3:	[0, 0, 0, 1] nonpromoter inactive
pozostałe to niejednoznaczne przyporzadkowania:
* 4-9 oznaczają dwa neurony wzbudzone:
	4:	[1, 1, 0, 0] aktywne
	5:	[1, 0, 1, 0] promotor
	6:	[1, 0, 0, 1]
	7:	[0, 1, 1, 0]
	8:	[0, 1, 0, 1] enhancer
	9:	[0, 0, 1, 1] nieaktywne
* 10-13 trzy neurony wzbudzone:
	10:	[1, 1, 1, 0]
	11:	[1, 1, 0, 1]
	12:	[1, 0, 1, 1]
	13:	[0, 1, 1, 1]
* 14,15 całkowicie niejasne:
	14:	[1, 1, 1, 1]
	15:	[0, 0, 0, 0]

"""


def kategory(wektor):
	wp=np.zeros((4))
	for k in range(4):
		if wektor[k]!=0.5:
			wp[k]=1
		else:
			wp[k]=0
	if sum(wp)==0:
		return 15
	elif sum(wp)==1:
		if wp[0]:
			return 0
		elif wp[1]:
			return 1
		elif wp[2]:
			return 2
		else:
			return 3
	elif sum(wp)==4:
		return 14
	elif sum(wp)==3:
		if not wp[0]:
			return 13
		elif not wp[1]:
			return 12
		elif not wp[2]:
			return 11
		else:
			return 10
	else:
		if wp[0]:
			if wp[1]:
				return 4
			elif wp[2]:
				return 5
			else:
				return 6
		elif wp[1]:
			if wp[2]:
				return 7
			else:
				return 8
		else:
			return 9



"""
funkcja w6(ile) tworzy wykresy słupkowe jednoznacznych i niejednoznacznych przyporządkowań 
(1 vs. wiele wzbudzonych) z podziałem na liczbę błędów w przyporządkowaniu;
patrz Rysunek 3.7
"""
def w6(ile):
	dane=p2(ile)
#	print(dane)
	fig,ax = plt.subplots(2,2,figsize=(20,20))
	fig.suptitle(f"sekwencje przyporządkowane jednoznaczenie przez >= {ile} sieci")
	fig.supylabel("liczebność")
	fig.supxlabel("jednoznaczność")
	axs=[ax[0,0],ax[0,1],ax[1,0],ax[1,1]]
	
	fig2,ax2 = plt.subplots(2,2,figsize=(20,20))
	fig2.suptitle(f"sekwencje przyporządkowane jednoznaczenie przez >= {ile} sieci",fontsize=24)
	fig2.supylabel("% w kategorii",fontsize=20)
	fig2.supxlabel("jednoznaczność",fontsize=20)	
	axs2=[ax2[0,0],ax2[0,1],ax2[1,0],ax2[1,1]]
	for kat in range(4):
		axs[kat].set_title(f"kategoria: {ki[kat]}",fontsize=20)
		axs2[kat].set_title(f"kategoria: {ki[kat]}",fontsize=20)
		axs2[kat].set_ylim(0,100)
		bottom = [0,0]
		prob=[sum(dane[kat,0]),sum(dane[kat,1])]
#		lab = [f"tak\n{prob[0]}",f"nie\n{prob[1]}"]			# wer1
		lab = [f"tak",f"nie"]								# wer2
		
		for r,d in enumerate(dane[kat].T):
			a=axs[kat].bar(lab, d,0.5, bottom=bottom, label=f"{r}")
			bottom=bottom+d
			if sum(d):
				axs[kat].bar_label(a, label_type='center')
		if not kat:
			fig.legend(title="liczba błędnych\nprzyporządkowań")
			
		bottom = np.zeros((2),dtype=float)
		x=np.zeros((2),dtype=float)
		for r,d in enumerate(dane[kat].T):
			for pi,pr in enumerate(prob):
				if pr:
					x[pi]=round(d[pi]/prob[pi]*100,2)
			a2=axs2[kat].bar(lab, x,0.5, bottom=bottom, label=f"{r}")
			axs2[kat].set_xticks(ticks=[0,1],labels=lab,fontsize=16)
			axs2[kat].set_yticks(ticks=[z*20 for z in range(0,6)],labels=[str(z*20) for z in range(0,6)],fontsize=16)
			bottom=bottom+x
			if sum(d):
				axs2[kat].bar_label(a2, label_type='center',fontsize=18,fmt='%.2f')
		if not kat:
			fig2.legend(title="liczba błędnych\nprzyporządkowań",fontsize=14,title_fontsize=15)
###			
#		a.set_title(lab,fontsize=20)
#		a.bar_label(v, label_type='center',fontsize=16)
####
	if out:
		fig.savefig(f"{out}{ff}_w6_{ile}_b.png")
		fig2.savefig(f"{out}{ff}_w6_{ile}_a.png")
		plt.close()
	else:
		plt.show()

"""
funkcja p2(ile) dla wszystkich modeli zwraca:
	* dane - macierz 4 x 2 x 5 zawierającą dla poszczególnych kategorii x 1 vs. wiele x liczba błędów [0-4]
	
	dla kryterium wyniki sieci są jednoznaczne jeśli są jednoznaczny dla >= ile modeli
"""
def p2(ile):
	dane =np.zeros((4,2,5),dtype=int)
	for kat in range(4):
		for l in range(lines): # 
			x = 0
			err=0
			for mod,neuron in enumerate(m4[l,kat]):
				if kategory(neuron)<4:
					x+=1
				if kat!=m2[l,kat,mod]:
					err+=1
			if x>=ile:
				y=0
			else:
				y=1
			dane[kat,y,err]+=1

	return dane
			

"""
funkcja s1(tnr) wypisuje do plików identyfikatory sekwencji, przyporządkownia i wyniki sieci dla modeli typu tnr:
1) błędne przyporządkowanie w tę samą stronę obu sekwencji [ 1 plik zmiana_p ]
2) błędne przyporządkowanie w przynajmniej przez 1 model [4 pliki zmaina_{prawdziwa}]

format 1):
id;	prawdziwa_kategoria; 	błędnia_kategoria;

format 2):
id;	prawdziwa_kategoria; 	błędnia_kategoria_1;	błędnia_kategoria_2;
"""

typy=["custom","alt","alt-again","ps_th2"]

def s1(tnr=0):
	print(f"analizuję {typy[tnr]}")
	if typ==11 and len(nazwy)==4:
		ff="/10k"
	elif typ==7:
		ff="/40k"
	else:
		ff=""													# dokałdna lokalizacja
	out1=open(f"zbiory{ff}/zmiana_k_{typy[tnr]}.txt","w+")
	out2_0=open(f"zbiory{ff}/zmiana_{typy[tnr]}_0.txt","w+")
	out2_1=open(f"zbiory{ff}/zmiana_{typy[tnr]}_1.txt","w+")
	out2_2=open(f"zbiory{ff}/zmiana_{typy[tnr]}_2.txt","w+")
	out2_3=open(f"zbiory{ff}/zmiana_{typy[tnr]}_3.txt","w+")
	outs=[out2_0,out2_1,out2_2,out2_3]
	if tnr==0:
		mod1=0
		mod2=1
	elif tnr==1:
		mod1=2
		mod2=3
	elif tnr==2:
		mod1=4
		mod2=5
	else:
		mod1=6
		mod2=7
	for kat in range(4):
		for l in range(lines):
			if kat==m2[l][kat][mod1]:
				if kat==m2[l][kat][mod2]:
					pass
				else:
					outs[kat].write(f"{m3[l][kat]}\t{kat}\t{kat}\t{m2[l][kat][mod2]}\t{m4[l][kat][mod1]}\t{m4[l][kat][mod2]}\n")

			else:
				if kat==m2[l][kat][mod2]:
					outs[kat].write(f"{m3[l][kat]}\t{kat}\t{m2[l][kat][mod1]}\t{kat}\t{m4[l][kat][mod1]}\t{m4[l][kat][mod2]}\n")
				else:
					outs[kat].write(f"{m3[l][kat]}\t{kat}\t{m2[l][kat][mod1]}\t{m2[l][kat][mod2]}\t{m4[l][kat][mod1]}\t{m4[l][kat][mod2]}\n")
					if m2[l][kat][mod2]==m2[l][kat][mod1]:
						out1.write(f"{m3[l][kat]}\t{kat}\t{m2[l][kat][mod1]}\t{m4[l][kat][mod1]}\t{m4[l][kat][mod2]}\n")
	
	out1.close()
	for plik in outs:
		plik.close()


"""
funkcja s2(ile) wypisyje do pliku sekwencjie, które pomyliły się w tę samą stronę  dla >=ile modeli

zwraca 1 plik zmana_{ile}.txt w formacie:

id;	prawdziwa_kategoria; kategoria_większości;	kat nazwy[0]; kat nazwy[1]; kat nazwy[2]; kat nazwy[3]
"""


def s2(ile):
	if typ==11 and len(nazwy)==4:
		ff="/10k"
	elif typ==7:
		ff="/40k"
	else:
		ff=""
	out=open(f"zbiory{ff}/zmiana_{ile}.txt","w+")
	for kat in range(4):
		for l in range(lines):
			rob=m2[l][kat]
			kand=[np.count_nonzero(rob == x) for x in range(4)]
			if kand.index(max(kand))!=kat and max(kand)>=ile:
				ind=kand.index(max(kand))
				w=""
				for iw in range(len(nazwy)):
					w=w+f"{m2[l][kat][iw]}\t"
				for iw in range(len(nazwy)):
					w=w+f"{kategory(m4[l,kat,iw,:])}\t"
				w=w.strip()
				out.write(f"{m3[l][kat]}\t{kat}\t{ind}\t{w}\n")
	out.close()


"""
funkcja s4(ile) wypisuje do plików identyfikatory sekwencji, przyporządkownia i wyniki sieci dla sekwencji 
przyporządkowanych poprawnie >=ile razy
do pliku stan_p.txt

format:
id;	prawdziwa_kategoria; 	błędnia_kategoria; 	kat nazwy[0, 1, ...]; kat_wekt[0, 1, ...]

"""

def s4(ile):
	print(f"próg poprawności >={ile}")
	if typ==11 and len(nazwy)==4:
		ff="/10k"
	elif typ==7:
		ff="/40k"
	else:
		ff=""													# dokałdna lokalizacja
	out=open(f"zbiory{ff}/stan_{ile}.txt","w+")
	for kat in range(4):
		for l in range(lines):
			rob=m2[l][kat]
			kand=[np.count_nonzero(rob == x) for x in range(4)]
			if kat==kand.index(max(kand)) and max(kand)>=ile:
				w=""
				for iw in range(len(nazwy)):
					w=w+f"{m2[l][kat][iw]}\t"
				for iw in range(len(nazwy)):
					w=w+f"{kategory(m4[l,kat,iw,:])}\t"
				w=w.strip()
				out.write(f"{m3[l][kat]}\t{kat}\t{w}\n")

	out.close()
	
"""
funkcja s5(ile) wypisuje do plików identyfikatory sekwencji, przyporządkownia i wyniki sieci dla sekwencji 
przyporządkowanych niejednoznacznie >=ile razy
do pliku niepewny_{ile}.txt

format:
id;	prawdziwa_kategoria; 	kat nazwy[0, 1, ...]; kat_wekt[0, 1, ...]

"""	
def s5(ile):
	print(f"próg niepewności przypasowania >={ile}")
	if typ==11 and len(nazwy)==4:
		ff="/10k"
	elif typ==7:
		ff="/40k"
	else:
		ff=""													# dokałdna lokalizacja
	out=open(f"zbiory{ff}/niepewny_{ile}.txt","w+")

	
	for kat in range(4):
		for l in range(lines):
			rob=m2[l][kat]
			typb=[0]*len(nazwy)
			czy=0
			for iw in range(len(nazwy)):
				typb[iw]=kategory(m4[l,kat,iw,:])
				if typb[iw] in [0,1,2,3]:
					czy+=1
			if czy<=len(nazwy)-ile:
				w=""
				for iw in range(len(nazwy)):
					w=w+f"{m2[l][kat][iw]}\t"
				for iw in typb:
					w=w+f"{iw}\t"
				w=w.strip()
				out.write(f"{m3[l][kat]}\t{kat}\t{w}\n")


	out.close()

"""
funkcja s6(ile) wypisuje do plików identyfikatory sekwencji i prawdziwe kategorie sieci dla sekwencji 
przyporządkowanych jednoznacznie >=ile razy do plików:

jendozn_{ile}_{kat}_{liczba błędnych przyporządkowań}.txt
niejednozn_{ile}_{kat}_{liczba błędnych przyporządkowań}.txt 
w formacie 
id;	prawdziwa kategoria


"""

def s6(ile):
	print(f"próg pewności przypasowania >={ile}")
	if typ==11 and len(nazwy)==4:
		ff="/10k"
	elif typ==7:
		ff="/40k"
	else:
		ff=""		

	
	for kat in range(4):
		out_j0=open(f"zbiory{ff}/n_jednozn/jednozn_{ile}_{kat}_0.txt","w+")
		out_n0=open(f"zbiory{ff}/n_jednozn/niejednozn_{ile}_{kat}_0.txt","w+")
		out_j1=open(f"zbiory{ff}/n_jednozn/jednozn_{ile}_{kat}_1.txt","w+")
		out_n1=open(f"zbiory{ff}/n_jednozn/niejednozn_{ile}_{kat}_1.txt","w+")
		out_j2=open(f"zbiory{ff}/n_jednozn/jednozn_{ile}_{kat}_2.txt","w+")
		out_n2=open(f"zbiory{ff}/n_jednozn/niejednozn_{ile}_{kat}_2.txt","w+")
		out_j3=open(f"zbiory{ff}/n_jednozn/jednozn_{ile}_{kat}_3.txt","w+")
		out_n3=open(f"zbiory{ff}/n_jednozn/niejednozn_{ile}_{kat}_3.txt","w+")
		out_j4=open(f"zbiory{ff}/n_jednozn/jednozn_{ile}_{kat}_4.txt","w+")
		out_n4=open(f"zbiory{ff}/n_jednozn/niejednozn_{ile}_{kat}_4.txt","w+")
		out_n=[out_n0,out_n1,out_n2,out_n3,out_n4]
		out_j=[out_j0,out_j1,out_j2,out_j3,out_j4]
		for l in range(lines): # 
			x = 0
			err=0
			for mod,neuron in enumerate(m4[l,kat]):
				if kategory(neuron)<4:
					x+=1
				if kat!=m2[l,kat,mod]:
					err+=1
			if x>=ile:
				fff=out_j[err]
				fff.write(f"{m3[l][kat]}\n")
			else:
				fff=out_n[err]
				fff.write(f"{m3[l][kat]}\n")

		for x in out_j:
			x.close()
		for x in out_n:
			x.close()

				
k=["promotor_active","nonpromotor_active", "promotor_inactive","nonpromotor_inactive"]
ki=["p.a.","np.a.","p.in.","np.in."]


"""
funkcja stat1() używa testu istotności z rozkładem dwumianowego,
wypisuje p-wartości dla 4 modeli obrazujące istotność hipotezy
h0 - uzyskane_błędy = oczekiwane_błędy
h1 - uzyskane_błędy != oczekiwane_błędy
h2 - uzyskane_błędy > oczekiwane_błędy

patrz Tabela A.1
"""
def stat1():
	print("Analiza 1 - rozkład dwumianowy\n{x}\nh1: błąd_uzyskany != błąd_oczekiwane\nh2: m_uzyskane > m_oczekiwane\n\n")
	for mod in range(len(nazwy)):
		print(f">>> testujemy sieć {nazwy[mod]}:\n")
		for kat in range(4):
			print(f"> kategoria: {k[kat]} 	prawd.={F_mod[mod,kat]}, % = {(lines-m1[kat,mod])/lines} ")
			r1 = binomtest(int(lines-m1[kat,mod]), n=lines, p=F_mod[mod,kat], alternative='two-sided')
			r2 = binomtest(int(lines-m1[kat,mod]), n=lines, p=F_mod[mod,kat], alternative='less')
			print(f"h1: p={r1.pvalue}\nh2: p={r2.pvalue}\n")


"""
funkcja stat2() używa dokładnego testu Fisher'a do sprawdzenia, czy dwa modele popełniają błędny 
w analogicznych miejscach z dla wszystkich par sieci, hipotezy:
h0 - zmienne niezależne
h1 - zmienne zależne

patrz Tabela A.2

ewentualnie przeprodwadza jeszcze test niezależności chi^2 oraz wylicza współczynnik zależności phi
"""
def stat2():
	print("Analiza 2 - test niezależności: dokładny test Fishera\n\nh1: zmienne zależne\n\n")
	for mod in range(len(nazwy)):
		for mod2 in range(mod):
			if mod!=mod2:
				print(f">>> testujemy sieć {nazwy[mod]} vs. {nazwy[mod2]}:\n")
				Fisher=p1(mod,mod2,1)

				for kat in range(4):
					print(f"> kategoria: {k[kat]} 	tablica zliczeń:\n ")
					print(Fisher[kat])
					fish=fisher_exact(Fisher[kat])
					print(f"Fisher 			p={fish.pvalue}")
#					chi2=chi2_contingency(Fisher[kat])
#					print(f"Chi2 z poprawką		p={chi2.pvalue}")
#					assoc=association(Fisher[kat],method='pearson')
#					print(f"Zależność met. Prearona	phi={assoc}\n")

"""
funkcja stat3() używa testu zgodności Chi^2 Pearsona do do sprawdzenia, czy dwa modele popełniają błędny 
w analogicznych miejscach dla wszystkich par sieci, hipotezy:
h0 - zmienne niezależne
h1 - zmienne zależne

patrz Tabela do Rysunku 3.4; github Stat_wykres_3_4.xlsx
"""
def stat3():
	print("Analiza 3 - test niezalezności: Chi^2 Pearsona\n\nh1: zmienne zależne\n\n")
	w=[]
	for mod in range(len(nazwy)):
		for mod2 in range(mod):
			if mod!=mod2:
				print(f">>> testujemy sieć {nazwy[mod]} vs. {nazwy[mod2]}:\n")
				kier, wk= p1(mod,mod2,4)
				w.append([wk,(mod,mod2)])
	for x in range(len(w)):
		for y in range(x,len(w)):
			if x!=y:

				print(f"Porównujemy rozkłady {nazwy[w[x][1][0]]} vs. {nazwy[w[x][1][1]]} kontra {nazwy[w[y][1][0]]} vs. {nazwy[w[y][1][1]]}")
				for kat in range(4):
					print(f"> kategoria: {k[kat]}\n ")
					x1=np.sum(np.reshape(w[x][0][kat],(16,1)))
					x2=np.sum(np.reshape(w[y][0][kat],(16,1)))
					xs=(x1+x2)/2
					obs=np.zeros((3,3))
					roz=np.zeros((3,3))
					xi=0
					for xx in range(4):
						yi=0
						if xx!=kat:
							for yy in range(4):
								if yy!=kat:
									obs[xi,yi]=w[x][0][kat][xx][yy]/x1*xs
									roz[xi,yi]=w[y][0][kat][xx][yy]/x2*xs
									yi+=1
							xi+=1
					chi_stat, chi_p=chisquare(np.reshape(obs,(9,1)),np.reshape(roz,(9,1)),ddof=0)
					print(f"chisquare	p={chi_p[0]}, chi={chi_stat[0]}\n")


"""
funkcja stat5(ile) używa testu U Manna-Whitney'a do porówania średnich, dla danych o ilości popełnionych
błędów przy przyporządkowywaniu sekwecji
z p2(ile), gdze ile - minimalna liczba poprawnych przyporzadkowań sekwencji przez sieci

patrz Tabela A.4
"""

def stat5(ile):
	dane=p2(ile)
	for kat in range(4):
		print(f"Badamy kategorię {k[kat]}")
		u_dane1=[0]*dane[kat][0][0]+[1]*dane[kat][0][1]+[2]*dane[kat][0][2]+[3]*dane[kat][0][3]+[4]*dane[kat][0][4]
		u_dane2=[0]*dane[kat][1][0]+[1]*dane[kat][1][1]+[2]*dane[kat][1][2]+[3]*dane[kat][1][3]+[4]*dane[kat][1][4]
		le=mannwhitneyu(u_dane1, u_dane2, method="auto",alternative="less")
		ts=mannwhitneyu(u_dane1, u_dane2, method="auto",alternative="two-sided")	
		gr=mannwhitneyu(u_dane1, u_dane2, method="auto",alternative="greater")
		print(f"M1 != M2	p={ts.pvalue}")	
		print(f"M1 < M2	p={le.pvalue}")
		print(f"M1 > M2	p={gr.pvalue}\n")
		x1=np.sum(dane[kat][0])
		x2=np.sum(dane[kat][1])
		x1=lines/x1
		x2=lines/x2
		chi_stat, chi_p=chisquare(dane[kat][0]*x1,dane[kat][1]*x2)
		print(f"chi		p={chi_p}\n")




f_list=[None for x in range(4)]

for x,f in enumerate(k):
	n=f"{path}/{f}.txt"
	n=n.replace("//","/")
	f_list[x]=open(n)



# główna pętla wczytująca dane wejściowe


for l in range(lines):
	for kat in range(4):
		line=f_list[kat].readline().split("\t")
		fun(line,kat,l)
		
if args.w1:
	w1()
if args.w2:
	w2()

if args.w3:
	if isinstance(args.w3, list):
		w=list(dict.fromkeys(args.w3))
		for para in w:
			para=para.strip().split("_")
			if int(para[0])<4 and int(para[0])>=0 and int(para[1])<4 and int(para[1])>=0:
				w3(int(para[0]),int(para[1]))
	else:
		para=para.strip().split("_")
		if int(para[0])<4 and int(para[0])>=0 and int(para[1])<4 and int(para[1])>=0:
			w3(int(para[0]),int(para[1]))	

if args.w4:
	if isinstance(args.w4, list):
		w=list(dict.fromkeys(args.w4))
		for para in w:
			para=para.strip().split("_")
			if int(para[0])<4 and int(para[0])>=0 and int(para[1])<4 and int(para[1])>=0:
				w4(int(para[0]),int(para[1]))
	else:
		para=para.strip().split("_")
		if int(para[0])<4 and int(para[0])>=0 and int(para[1])<4 and int(para[1])>=0:
			w4(int(para[0]),int(para[1]))
if args.w5:
	if isinstance(args.w5, list):
		w=list(dict.fromkeys(args.w5))
		for para in w:
			if para>=0 and para<=4:
				w6(para)
	else:
		w6(args.w5)
if args.s1:
	if isinstance(args.s1, list):
		w=list(dict.fromkeys(args.s1))
		for para in w:
			s1(para)
	else:
		s1(args.s1)
if args.s2:
	if isinstance(args.s2, list):
		w=list(dict.fromkeys(args.s2))
		for para in w:
			s2(para)
	else:
		s2(args.s2)
if args.s3:
	if isinstance(args.s3, list):
		w=list(dict.fromkeys(args.s3))
		for para in w:
			s4(para)
	else:
		s4(args.s3)
if args.s4:
	if isinstance(args.s4, list):
		w=list(dict.fromkeys(args.s4))
		for para in w:
			s5(para)
	else:
		s5(args.s4)
if args.s5:
	if isinstance(args.s5, list):
		w=list(dict.fromkeys(args.s5))
		for para in w:
			s6(para)
	else:
		s6(args.s5)

if args.stat1:
	stat1()
if args.stat2:
	stat2()
if args.stat3:
	stat3()
if args.stat4:
	stat5(args.stat4)



for f in f_list:
	f.close()
