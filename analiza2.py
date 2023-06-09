"""
zbiór fukcji do graficznej (na razie) analizy przefiltrowanych plików z id i kategorioami
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import mannwhitneyu,chisquare

nazwy=["custom40", "custom41", "alt1", "alt2"]
ki=["p.a.","np.a.","p.in.","np.in."]
ks=["pa","na","pi","ni"]
ko=["blue","darkorange","green","red"]
typy=["custom","alt"]
col=["r","r","r","r","g","g","lime","lime","g","g","b","b","b","b","c","blueviolet"]
lab=["0: pr-a", "1: nonpr-a", "2: pr-in","3: nonpr-in","4: a","5: pr","6: pr-a/nonpr-in","7: nonpr-a/pr-a","8: nonpr", "9: in","10: pr-a/nonpr-a/pr-in","11: pr-a/nonpr-a/nonpr-in","12: pr-a/pr-in/nonpr-in","13: nonpr-a/pr-in/nonpr-in","14: all", "15: non"]


"""
parsowanie argumentów
"""

parser = argparse.ArgumentParser(description=f'Wstępna analiza danych z {len(nazwy)} modeli: {nazwy}',formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-p','--path', nargs=1, help="""Ścieżka prowadząca do katalogu z plikami zawierającymi piki do podsumowania, np.:
-> zmiana_{int}\n-> zmiana_{rodzaj modelu}_{int}\n-> zmiana_k_{rodzaj modelu}\n-> zmiana_r_{int}_{int}\n-> stan_{int}\n-> niepewny_{ile}""",default=os.getcwd())
parser.add_argument('-p4','--path4', nargs=1, help="""Ścieżka prowadząca do katalogu z plikami zawierającymi piki z id sekwencji oraz poziomem metylacji i acetylacji, tzn.:
-> id_met.txt\n-> id_ac.txt\nniezbędne do analizy 4 i 5\n""",default=os.getcwd())
parser.add_argument('-o','--out', nargs=1, help="Ścieżka prowadząca do katalogu, w którym program am zapisać wykresy; w przypadku braku tylko pokazuje",default=None)
parser.add_argument('-m0', 
help="""analiza zmiana_{int} - typy błedów, kierunkowe przypisanie do błędnej kategorii dla progów 2,3,4""",
action='store_true', default=False)

parser.add_argument('-bins',  nargs='?', const=26, type=int,
help="""liczba przedziałów dla histogramów przy -m1, -m3, -m5, domyślnie 26;""", default=26)

parser.add_argument('-m1',action='store_true',
help="""histogram metylacji \ acetylacji dla błędów: p.a <-> p.ia oraz np.a. <-> p.a.;
wykorzystuje pliki zmaina_{ile};
M1 - liczba przedziałów, domyślnie 26;""", default=False)

parser.add_argument('-m2', 
help="""analiza stan_{int} - wykres typów błędów większościowo poprawnie przyporządkowanych sekwencji""",
action='store_true', default=False)

parser.add_argument('-m3', action='store_true',
help="""histogram metylacji \ acetylacji dla poprawnie przyporządkowanych sekwencji;
wykorzystuje pliki zmaina_{ile};
M1 - liczba przedziałów, domyślnie 26;""", default=False)

parser.add_argument('-m4', 
help="""analiza sekwencji o wielokrkotnie wzbudzonych neuronach;
wykorzystuje niepewny_{ile}""",
action='store_true', default=False)

parser.add_argument('-m5', nargs="+",
help="""porównanie rozkładów metylacji i acetylacji dla błędów typu  p.a <-> p.in vs. poprawne rozkłady;
wykorzystuje stan_{ile} oraz zmiana_{ile}
wprowadzanie typu zamiany w postaci: 0_2 zamiana p.a. -> p.in""", default=False)

parser.add_argument('-n', help="""czy operujemy na średnich poziomach metylacji/acetylacji znaormalizowanych; 
domyślnie True, pojawienie się flagi zmiena na False""",action='store_false', default=True)
parser.add_argument('-s', help="""test U Manna-Whitney'a do porówania przesunięć rozkładów metylacji i acetylacji;
wykonywany tylko razem z analizą -m 9""",action='store_true', default=False)
parser.add_argument('-sam', help="""czy zapisywać id i poziomy metylacji i acetylacji badanych podzbiorów;
wykonywane tylko wraz z 4,5,6,7,9""",action='store_true', default=False)

args = parser.parse_args()

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
path4=args.path4[0]

global ff

if "3" in path4:
	ff="10k_"
elif "6" in path4:
	ff="40k_"
else:
	ff=""

global norm
global s_am
global stat
norm = args.n
s_am=args.sam
stat=args.s

# kategorie, których id i poziomy met /ac wypisać do plików przy -sam
global k_am
k_am=[0,1,2]
# błędy kierunkowe, których id i poziomy met /ac wypisać do plików przy -sam
global b_am
b_am=[(0,1),(1,0),(0,2),(2,0)]

"""
sprawdzanie poprawności ścieżki path i 

"""
if os.getcwd() == path:
	print("Podana ścieżka wskazuje n abieżący katalog")
elif len(os.path.commonpath([os.getcwd(),path]))>1:
	print("Podana ścieżka wskazuje na istniejący katalog")
else:
	path=os.getcwd()+f"/{path}"
	print(f"operujemy w katalogu {path}")
	if not os.path.exists(path):
		print("podana ścieżka do katalogu z danymi jest niepoprawna")

if args.m3 or args.m1:

	if os.getcwd() == path4:
		print("Podana ścieżka wskazuje na abieżący katalog")
	elif len(os.path.commonpath([os.getcwd(),path4]))>=len(os.getcwd()):
		print("Podana ścieżka wskazuje na istniejący katalog")
	else:
		path4=os.getcwd()+f"/{path4}"
		print(f"szukamy metylacji i acetylacji w katalogu {path4}")
	if not os.path.exists(path4):
		print("podana ścieżka do katalogu z danymi jest niepoprawna")
	if "dataset3" in path4:
		ds=3
	elif "dataset6" in path4:
		ds=6
	else:
		print("podaj poprwany katalog z danymi o metylacji i acetylacji")
	if norm:
		met=f"{path4}/ds{ds}_id_met_norm.txt"
		ac=f"{path4}/ds{ds}_id_ac_norm.txt"
	else:
		ac=f"{path4}/ds{ds}_id_ac.txt"
		met=f"{path4}/ds{ds}_id_met.txt"
	met=met.replace("//","/")		
	ac=ac.replace("//","/")


"""
funkcja wczytaj_03(p_file) wczytuje do dane ze ścieżki p_file w formacie:

id;	prawdziwa_kategoria; kategoria_większości;	kat_przp[0,1,2,4] typ_błędu[1, 2, ... 16]
"""
def wczytaj_03(p_file):
	m5=np.zeros((4,4),dtype=int)
	m6=np.zeros((4,16),dtype=int)
	with open(p_file,"r") as pf:
		for line in pf:
			line=line.strip().split("\t")
			m5[int(line[1])][int(line[2])]+=1
			for k in range(3+int((len(line)-3)/2),len(line)):
				m6[int(line[1])][int(line[k])]+=1
			
	return m5,m6

"""
funkcja wczytaj_1(p_file) wczytuje do dane ze ścieżki p_file w formacie:

id;	prawdziwa_kategoria; 	błędnia_kategoria_1;	błędnia_kategoria_2;
"""

def wczytaj_1(p_file):
	mp1=np.zeros(3)			# 1 błąd; 2 takie same; 2 różne
	mp2=np.zeros(4)			# liczba błednych przyporządkowań do danych kategorii
	with open(p_file,"r") as pf:
		for line in pf:
			line=line.strip().split("\t")
			if line[1]==line[2]:
				mp1[0]+=1
				mp2[int(line[3])]+=1
			else:
				mp2[int(line[2])]+=1
				if line[1]==line[3]:
					mp1[0]+=1
				else:
					mp2[int(line[3])]+=1
					if line[2]==line[3]:
						mp1[1]+=1
					else:
						mp1[2]+=1
		
	return mp1,mp2

"""
funkcja wczytaj_1(p_file) wczytuje do dane ze ścieżki p_file w formacie:

id;	prawdziwa_kategoria; 	błędnia_kategoria;
"""	
def wczytaj_2(p_file):
	mp=np.zeros((4,4),dtype=int)
	with open(p_file,"r") as pf:
		for line in pf:
			line=line.strip().split("\t")
			mp[int(line[1])][int(line[2])]+=1
	return mp
	
"""


funkcja wczytaj_sl(ac, met) wczytuje słowniki metylacji i acetylacji z katalogu path4

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
			dict_ac[line[0]]=(line[1],line[2])		# czy potrzeba prawdziwej kategorii?
	with open(met,"r") as pmet:
		for line in pmet:
			line=line.strip().split("\t")
			dict_met[line[0]]=(line[1],line[2])

"""
funkcja wczytaj_45(p_file) wczytuje dane o metylacji i acetylacji sekwencji
z pliku ze ścieżki p_file w formacie:

id;	prawdziwa_kategoria; kategoria_większości;	kat_przp[0,1,2,4] typ_błędu[1, 2, ... 16]

oraz dane o metylacji i acetylacji z katalogu path4

format słownika m5:
	{prawdziwa_kat : {przypisana_kat : [[met,typ_bł],[ac,typ_bł]] } }
	
m6 zpisuje braki w danych o metylacji / acetylacji sekwencji w formacie:
	2 x 4 x 4 -> met/ac x prawdziwa_kat x przypisana_kat
	

"""




def wczytaj_45(p_file):
	m5={0:{1:[[],[]],2:[[],[]],3:[[],[]]},1:{0:[[],[]],2:[[],[]],3:[[],[]]},2:{1:[[],[]],0:[[],[]],3:[[],[]]},3:{1:[[],[]],0:[[],[]],2:[[],[]]}}
	m6=np.zeros((2,4,4),dtype=int)
	# istnieją już słowniki metylacji i acetylacji

	# przy załóżeniu, że id się nie powtarzają między grupami:
	if s_am:
		pn=f"{path}/am/"
		pn=pn.replace("//","/")
		out=[None] * len(b_am)
		if "2" in p_file:
			ile=2
		elif "3" in p_file:
			ile=3
		elif "4" in p_file:
			ile=4
		else:
			print("zmień ustawienia dla większościowego przyporządkowania, ile poza obsługiwanym zbiorem {2,3,4}")
			ile=0
		for zi,zm in enumerate(b_am):
			fff=open(f"{pn}błąd_{ile}_{ks[zm[0]]}_{ks[zm[1]]}.txt","w+")
			out[zi]=fff

	with open(p_file,"r") as pf:
		for line in pf:
			line=line.strip().split("\t")
			if line[0] in dict_met.keys():
				b=[0,0,0,0,0]
				b[0]=round(float(dict_met[line[0]][0]),2)
				for x in range(1,len(nazwy)+1):
					b[x]=int(line[x+2+int((len(line)-3)/2)])
				m5[int(line[1])][int(line[2])][0].append(b)
				if s_am:
					for xi,x in enumerate(b_am):	
						if int(line[1])==x[0] and x[1]==int(line[2]):
							out[xi].write(f"{line[0]}\t{dict_met[line[0]][0]}\t")
			else:
				m6[0][int(line[1])][int(line[2])]+=1
			if line[0] in dict_ac.keys():
				
				b=[0,0,0,0,0]
				b[0]=round(float(dict_ac[line[0]][0]),2)
				for x in range(1,len(nazwy)+1):
					b[x]=int(line[x+2+int((len(line)-3)/2)])
				m5[int(line[1])][int(line[2])][1].append(b)
				if s_am:
					for xi,x in enumerate(b_am):	
						if int(line[1])==x[0] and x[1]==int(line[2]):
							out[xi].write(f"{dict_ac[line[0]][0]}\n")
			else:
				m6[1][int(line[1])][int(line[2])]+=1
	if s_am:
		for xi,x in enumerate(b_am):
			out[xi].close()
	return m5,m6

"""
funkcja wczytaj_stan(p_file) wczytuje dane o metylacji i acetylacji sekwencji
z pliku ze ścieżki p_file w formacie:

id;	prawdziwa_kategoria; 	kat_przp[0,1,2,4] typ_błędu[1, 2, ... 16]

oraz dane o metylacji i acetylacji z katalogu path4

format słownika m5:
	{prawdziwa_kat : {przypisana_kat : [[met,typ_bł],[ac,typ_bł]] } }
	
m6 zpisuje braki w danych o metylacji / acetylacji sekwencji w formacie:
	2 x 4 x 4 -> met/ac x prawdziwa_kat x przypisana_kat
	

"""

def wczytaj_stan(p_file):
	m5={0:{0:[[],[]],1:[[],[]],2:[[],[]],3:[[],[]]},1:{0:[[],[]],1:[[],[]],2:[[],[]],3:[[],[]]},2:{1:[[],[]],0:[[],[]],2:[[],[]],3:[[],[]]},3:{1:[[],[]],0:[[],[]],3:[[],[]],2:[[],[]]}}
	m6=np.zeros((2,4,4),dtype=int)
	# istnieją już słowniki metylacji i acetylacji

	# przy załóżeniu, że id się nie powtarzają między grupami:
	if s_am:
		pn=f"{path}/am/"
		pn=pn.replace("//","/")
		out=[None] * 4
		if "2" in p_file:
			ile=2
		elif "3" in p_file:
			ile=3
		elif "4" in p_file:
			ile=4
		else:
			print("zmień ustawienia dla większościowego przyporządkowania, ile poza obsługiwanym zbiorem {2,3,4}")
			ile=0
		for zm in k_am:
			fff=open(f"{pn}poprawne_{ile}_{ks[zm]}.txt","w+")
			out[zm]=fff
	with open(p_file,"r") as pf:
		for line in pf:
			line=line.strip().split("\t")
			if line[0] in dict_met.keys():
				b=[0,0,0,0,0]
				b[0]=round(float(dict_met[line[0]][0]),2)
				for x in range(len(nazwy)):
					b[x+1]=int(line[x+2+int((len(line)-2)/2)])
				m5[int(line[1])][int(line[2])][0].append(b)
				if s_am:
					if int(line[1])==int(line[2]) and int(line[1]) in k_am:
						out[int(line[1])].write(f"{line[0]}\t{dict_met[line[0]][0]}\t")
			else:
				m6[0][int(line[1])][int(line[2])]+=1
			if line[0] in dict_ac.keys():
				
				b=[0,0,0,0,0]
				b[0]=round(float(dict_ac[line[0]][0]),2)
				for x in range(len(nazwy)):
					b[x+1]=int(line[x+2+int((len(line)-2)/2)])
				m5[int(line[1])][int(line[2])][1].append(b)
				if s_am:
					if int(line[1])==int(line[2]) and int(line[1]) in k_am:
						out[int(line[1])].write(f"{dict_ac[line[0]][0]}\n")
			else:
				m6[1][int(line[1])][int(line[2])]+=1
	if s_am:
		for x in k_am:
			out[x].close()
	return m5,m6



"""
funkcja wczytaj_niepewny(p_file) wczytuje do dane ze ścieżki p_file w formacie:

id;	prawdziwa_kategoria;	kat_przp[0,1,2,4] typ_błędu[1, 2, ... 16]

m5 - macierz zawierająca liczebności poszczególnych przypisań do kategorii dla plików niepweny_{ile}
		4 x 4 -> prawdziwa_kategoria x przypisana kategoria
"""
def wczytaj_niepewny(p_file):
	m5=np.zeros((4,4),dtype=int)
	m6=np.zeros((4,16),dtype=int)
	with open(p_file,"r") as pf:
		for line in pf:
			line=line.strip().split("\t")
			for k in range(2+int((len(line)-2)/2),len(line)):
				m5[int(line[1])][int(line[k-4])]+=1
				m6[int(line[1])][int(line[k])]+=1
			
	return m5,m6
	
####### funkcje pomocnicze


"""
funkcja stat4(roz1, roz2) używa testu U Manna-Whitney'a doporówania średnich dla podanych rozkładów 

"""

def stat4(roz1,roz2):
	le=mannwhitneyu(roz1, roz2, method="auto",alternative="less")
	ts=mannwhitneyu(roz1, roz2, method="auto",alternative="two-sided")	
	gr=mannwhitneyu(roz1, roz2, method="auto",alternative="greater")
	print(f"M1 != M2	p={ts.pvalue}")	
	print(f"M1 < M2	p={le.pvalue}")
	print(f"M1 > M2	p={gr.pvalue}\n")



"""
funkcja w1(m5, m6, k, zmiana) tworzy wkresy dla analiz -m1, -m3, -m5:
	> m5,m6 - dane
	> k 	- wybór analizy 0,4,7,9
	> zmaiana obejmuje:
		> z		- prawdziwa kategoria, którą źle dopasowano 
		> do	- błędna kategoria, do kßórej przyporządkowano większość
		> ile   - próg przyporządkowania w danym kierunku
"""			
def w1(m5,m6,k, zmiana):
	fig,ax = plt.subplots(1,2,figsize=(20,10))
	m_ys=np.array(m5[zmiana[0]][zmiana[1]][0])
	m_xs=m_ys[:,0]
	m_ys=m_ys[:,1:]
	a_ys=np.array(m5[zmiana[0]][zmiana[1]][1])
	a_xs=a_ys[:,0]
	a_ys=a_ys[:,1:]
	
	ax[0].set_title(f"rozkład metylacji",fontsize=20)	# lub  \nbraki w danych: {m6[0][zmiana[0]][zmiana[1]]} na {m6[0][zmiana[0]][zmiana[1]]+len(m5[zmiana[0]][zmiana[1]][0])}
	ax[1].set_title(f"rozkład acetylacji",fontsize=20)	# lub  \nbraki w danych: {m6[1][zmiana[0]][zmiana[1]]} na {m6[1][zmiana[0]][zmiana[1]]+len(m5[zmiana[0]][zmiana[1]][1])}
	if norm:	
		nn="\ndane znormalizowane"
	else:
		nn="\ndane nie-znormalizowane"
	if k==4:
		title=f"w4_{ks[zmiana[0]]}_{ks[zmiana[1]]}_{zmiana[2]}"
		fig.suptitle(f"rozkład modyfikacji sekwencji dla błędu:\nzamiana {ki[zmiana[0]]} -> {ki[zmiana[1]]}\nprzy kierunkowym błędzie >={zmiana[2]}{nn}")
		fig.supylabel("liczebnośc")
		ax[0].set_xlabel("poziom metylacji")
		ax[1].set_xlabel("poziom acetylacji")
		ax[0].hist(m_xs,args.bins)
		ax[1].hist(a_xs,args.bins)

	elif k==7:
		title=f"w7_{zmiana[0]}_1"
		fig.suptitle(f"rozkład modyfikacji sekwencji dla poprawnego przyporządkowania>={zmiana[2]}\ndla kategorii {ki[zmiana[0]]}{nn}")
		fig.supylabel("liczebność")
		ax[0].set_xlabel("poziom metylacji")
		ax[1].set_xlabel("poziom acetylacji")
		ax[0].hist(m_xs,args.bins)
		ax[1].hist(a_xs,args.bins)
	elif k==0:
		title=f"w7_{zmiana[0]}_2"
		bottom0=[0]*args.bins
		bottom1=[0]*args.bins
		fig.suptitle(f"rozkład metylacji i acetylacji sekwencji dla błędnego przyporządkowania<={len(nazwy)-zmiana[2]}\ndla kategorii {ki[zmiana[0]]}{nn}")	# dla kategorii {ki[zmiana[0]]}
		fig.supylabel("liczebność")
		l=[0,1,2,3]
		l.pop(zmiana[0])
		b0,b1=np.zeros((1)),np.zeros((1))
		ax[0].set_xlabel("poziom metylacji")
		ax[1].set_xlabel("poziom acetylacji")
		for x in l:
			m_ys=np.array(m5[zmiana[0]][x][0])
			a_ys=np.array(m5[zmiana[0]][x][1])
			if m_ys.size>0:
				m_xs=m_ys[:,0]
				if b0.any():
					n0,b0,p0 = ax[0].hist(m_xs,b0,bottom=bottom0,label=ki[x],color=ko[x]) # [0,10,20,40,80,max(m_xs)]	,density=True
				else:
					n0,b0,p0 = ax[0].hist(m_xs,args.bins,bottom=bottom0,label=ki[x],color=ko[x])
				bottom0=bottom0+n0
			if a_ys.size>0:
				a_xs=a_ys[:,0]
				if b1.any():	
					n1,b1,p1 = ax[1].hist(a_xs,b1,bottom=bottom1,label=ki[x],color=ko[x])
				else:
					n1,b1,p1 = ax[1].hist(a_xs,args.bins,bottom=bottom1,label=ki[x],color=ko[x])
				bottom1=bottom1+n1
			plt.legend(fontsize=14)
	elif k==9:
		
		title=f"w9_{ks[zmiana[0]]}_{ks[zmiana[1]]}_{zmiana[2]}_{args.bins}"
		fig.suptitle(f"rozkłady poziomów modyfikacji\nzamiana {ki[zmiana[0]]} -> {ki[zmiana[1]]}",fontsize=24)
#		fig.suptitle(f"porównanie rozkładów modyfikacji sekwencji\n dla błedu kierunkowego/poprawności >={zmiana[2]} \nzamiana {ki[zmiana[0]]} -> {ki[zmiana[1]]}{nn}")
		fig.supylabel("gęstość",fontsize=20)
		
		xa_max=max(a_xs)
		xm_max=max(m_xs)
		xa_min=min(a_xs)
		xm_min=min(m_xs)
		
		m_ys=np.array(m6[zmiana[0]][zmiana[0]][0])
		m_xs=m_ys[:,0]
		a_ys=np.array(m6[zmiana[0]][zmiana[0]][1])
		a_xs=a_ys[:,0]

		xa_max=max(max(a_xs), xa_max)
		xm_max=max(max(m_xs), xm_max)
		xa_min=min(min(a_xs), xa_min)
		xm_min=min(min(m_xs), xm_min)
		
		m_ys=np.array(m6[zmiana[1]][zmiana[1]][0])
		m_xs=m_ys[:,0]
		a_ys=np.array(m6[zmiana[1]][zmiana[1]][1])
		a_xs=a_ys[:,0]
		
		xa_max=max(max(a_xs), xa_max)
		xm_max=max(max(m_xs), xm_max)
		xa_min=min(min(a_xs), xa_min)
		xm_min=min(min(m_xs), xm_min)
		
		m_ys=np.array(m5[zmiana[0]][zmiana[1]][0])
		m_xs=m_ys[:,0]
		a_ys=np.array(m5[zmiana[0]][zmiana[1]][1])
		a_xs=a_ys[:,0]
		
		binm=np.linspace(xm_min,xm_max, args.bins)
		bina=np.linspace(xa_min,xa_max, args.bins)
		bb_m=[round((binm[x]+binm[x+1])/2,3) for x in range(len(binm)-1)]
		bb_a=[round((bina[x]+bina[x+1])/2,3) for x in range(len(bina)-1)]
		
		d_1m=m_xs
		d_1a=a_xs
		
		ax[0].set_xlabel("poziom metylacji",fontsize=20)
		ax[1].set_xlabel("poziom acetylacji",fontsize=20)
		c_1m, b_1m, p = ax[0].hist(m_xs,binm,label=f"{ki[zmiana[0]]} -> {ki[zmiana[1]]}",density=True,color="C4",alpha=0.75) # [0,10,20,40,80,max(m_xs)]	,density=True
		c_1a, b_1a, p = ax[1].hist(a_xs,bina,label=f"{ki[zmiana[0]]} -> {ki[zmiana[1]]}",density=True,color="C4",alpha=0.75)
		

		m_ys=np.array(m6[zmiana[0]][zmiana[0]][0])
		m_xs=m_ys[:,0]
		a_ys=np.array(m6[zmiana[0]][zmiana[0]][1])
		a_xs=a_ys[:,0]
#		bina=[-10,0.5,1,2,10,30]
		c_2m, b_1m, p = ax[0].hist(m_xs,binm,label=f"{ki[zmiana[0]]}",density=True,color=f"C{zmiana[0]}",histtype="step",lw=3) # [0,10,20,40,80,max(m_xs)]	,density=True
		c_2a, b_1a, p = ax[1].hist(a_xs,bina,label=f"{ki[zmiana[0]]}",density=True,color=f"C{zmiana[0]}",histtype="step",lw=3) # density=True, 
					# przy liczeniu % niskiego poziomu usuń density = True
		if stat:
			print(f">> U Manna-Whitney porównanie {ki[zmiana[0]]} -> {ki[zmiana[1]]} vs {ki[zmiana[0]]}:")
			print("metylacja:\n")
			stat4(d_1m,m_xs)
			print("acetylacja:\n")
			stat4(d_1a,a_xs)

	
		m_ys=np.array(m6[zmiana[1]][zmiana[1]][0])
		m_xs=m_ys[:,0]
		a_ys=np.array(m6[zmiana[1]][zmiana[1]][1])
		a_xs=a_ys[:,0]
		
		c_3m, b_1m, p = ax[0].hist(m_xs,binm,label=f"{ki[zmiana[1]]}",density=True,color=f"C{zmiana[1]}",histtype="step",lw=3) # [0,10,20,40,80,max(m_xs)]	,density=True
		c_3a, b_1a, p = ax[1].hist(a_xs,bina,label=f"{ki[zmiana[1]]}",density=True,color=f"C{zmiana[1]}",histtype="step",lw=3)
		ax[0].tick_params(axis='both', which='major', labelsize=16)
		ax[1].tick_params(axis='both', which='major', labelsize=16)
		plt.legend(title="rozkład",fontsize=16,title_fontsize=18)
		
		if stat:
			print(f">> U Manna-Whitney porównanie {ki[zmiana[0]]} -> {ki[zmiana[1]]} vs {ki[zmiana[1]]}:")
			print("metylacja:\n")
			stat4(d_1m,m_xs)
			print(f"Procentowy udział niskiego poziomu: {c_3a[0]/c_3a.sum()} <= {b_1a[1]}")
			stat4(d_1a,a_xs)


		
				
	if norm:
		title=title+"n"
	if out:
		fig.savefig(f"{out}{ff}{title}.png")
		plt.close()
	else:		
		plt.show()				

"""
funkcja met_ac(path_) sprawdza, czy w katalogu path_ istnieją pliki z danymi o metylacji i acetylacji 
"""				
def met_ac (path_):
	if os.getcwd() == path_:
		print("Podana ścieżka wskazuje na bieżący katalog")
	elif len(os.path.commonpath([os.getcwd(),path_]))>=len(os.getcwd()):
		print("Podana ścieżka wskazuje na istniejący katalog")
	else:
		path_=os.getcwd()+f"/{path_}"
		print(f"szukamy metylacji i acetylacji w katalogu {path_}")
	if not os.path.exists(path_):
		print("podana ścieżka do katalogu z danymi jest niepoprawna")
	if "dataset3" in path_:
		ds=3
	elif "dataset6" in path_:
		ds=6
	else:
		print("podaj poprwany katalog z danymi o metylacji i acetylacji")
		return None
	if norm:
		met=f"{path_}/ds{ds}_id_met_norm.txt"
		ac=f"{path_}/ds{ds}_id_ac_norm.txt"
	else:
		ac=f"{path_}/ds{ds}_id_ac.txt"
		met=f"{path_}/ds{ds}_id_met.txt"
	met=met.replace("//","/")
	ac=ac.replace("//","/")
	if os.path.isfile(met) and os.path.isfile(ac):
		return (met, ac)
	else:
		print(f"Brak pliku {met} lub {ac}")
		return None


#### analiza

if args.m0:
	razem=[]
	xs=[]
	for ile in range(2,len(nazwy)+1):
		n=f"{path}/zmiana_{ile}.txt"
		n=n.replace("//","/")
		if os.path.isfile(n):				
			m5,m6=wczytaj_03(n)
			ylim=max(m5.sum(1))
			xs.append(ile)
			razem.append(m5)
			######################################################
			fig,ax = plt.subplots(2,2,figsize=(20,20))
			axs=[ax[0,0],ax[0,1],ax[1,0],ax[1,1]]
			fig.suptitle(f"typy błędów\ndla sekwencji przypocządkowanych do tej samej błędnej kategorii >={ile} razy\nmodele: {nazwy}")
			fig.supxlabel("typ")
			fig.supylabel("liczba sekwencji")
			for ai,a in enumerate(axs):
				a.set_title(f"prawdziwa kategoria: {ki[ai]} czyli {ai}\nliczba błędów {sum(m6[ai,:])-m6[ai,ai]}, liczba poprawnych: {m6[ai,ai]}")
				col[ai]="orange"
				if ai:
					col[ai-1]="r"
				if not ai:
					aa=a.bar(range(16),m6[ai,:],color=col,label=lab)
				else:
					aa=a.bar(range(16),m6[ai,:],color=col)
				a.bar_label(aa)
			fig.legend(loc='outside upper right')
			if out:
				fig.savefig(f"{out}{ff}w0_2_{ile}.png")
			else:
				plt.show()
			col[3]='r'
			######################################################
			fig,ax =plt.subplots(figsize=(10,10))
			fig.supylabel("liczba błędów")
			fig.supxlabel("prawdziwa kategoria")
			ax.set_title(f"Sekwencje przypisane do tej samej błędnej kategorii >= {ile} razy")
			ax.set_ylim(0,ylim)
			bottom=[0,0,0,0]
			for r,d in enumerate(m5.T):
				a=ax.bar(ki, d,0.5, bottom=bottom, label=f"{ki[r]}")
				bottom=bottom+d
				ax.bar_label(a,)		# label_type='center'
					
			fig.legend(title="przypisana kategoria")
				
			if out:
				fig.savefig(f"{out}{ff}w0_0_{ile}.png")
			else:
				plt.show()
		else:
			print(f"plik zmiana_{ile} nie istnieje")
	###############################################################		
	fig,ax =plt.subplots(2,2,figsize=(10,10))
	axs = [ax[0,0],ax[0,1],ax[1,0],ax[1,1]]
	fig.supylabel("liczba błędnych przyporządkowań",fontsize=20)
	fig.suptitle("Sekwencje przypisane do tej samej błędnej kategorii >= x",fontsize=24)
	fig.supxlabel("x",fontsize=20)
	ys=np.zeros((len(xs),4),dtype=int)
			
	for ai,a in enumerate(axs):
		for y in range(len(razem)):
			ys[y,:]=razem[y][ai]
		ylim=max(ys.sum(1))
		a.set_ylim(0,ylim)
		a.stackplot(xs,ys.T, labels=ki, alpha=0.8)
		a.tick_params(axis='both', which='major', labelsize=14)
		a.tick_params(axis='both', which='minor', labelsize=14)
		a.set_title(f'prawdziwa kategoria: {ki[ai]}',fontsize=16)
		a.set_xticks(xs)
		if ai==1:
			a.legend(loc='upper right',fontsize=16,title_fontsize=18)
	if out:
		fig.savefig(f"{out}{ff}w0_1.png")
	else:
		plt.show()
		
if args.m1:
	if os.path.isfile(met) and  os.path.isfile(ac):
		wczytaj_sl(ac,met)	
		for ile in range(2,len(nazwy)+1):
			n=f"{path}/zmiana_{ile}.txt"
			n=n.replace("//","/")
			if os.path.isfile(n):
					
				# p. a. -> p. in.	
				m5,m6=wczytaj_45(n)
				w1(m5,m6,4,[0,2,ile])
				w1(m5,m6,4,[2,0,ile])
				w1(m5,m6,4,[1,0,ile])
				w1(m5,m6,4,[0,1,ile])
	else:
		print("brakuje pliku id_met.txt lub id_ac.txt")	

if args.m2:
	razem=[]
	xs=[]
	for ile in range(2,len(nazwy)+1):
		n=f"{path}/stan_{ile}.txt"
		n=n.replace("//","/")
		if os.path.isfile(n):				
			m5,m6=wczytaj_03(n)
			ylim=max(m5.sum(1))
			xs.append(ile)
			razem.append(m5)
			######################################################
			fig,ax = plt.subplots(2,2,figsize=(20,20))
			axs=[ax[0,0],ax[0,1],ax[1,0],ax[1,1]]
			fig.suptitle(f"typy błędów\ndla sekwencji przypocządkowanych poprawnie >={ile} razy\nmodele: {nazwy}")
			fig.supxlabel("typ")
			fig.supylabel("liczba sekwencji")
			for ai,a in enumerate(axs):
				a.set_title(f"prawdziwa kategoria: {ki[ai]} czyli {ai}\nliczba błędów {sum(m6[ai,:])-m6[ai,ai]}, liczba poprawnych: {m6[ai,ai]}")
				col[ai]="orange"
				if ai:
					col[ai-1]="r"
				if not ai:
					aa=a.bar(range(16),m6[ai,:],color=col,label=lab)
				else:
					aa=a.bar(range(16),m6[ai,:],color=col)
				a.bar_label(aa)
			fig.legend(loc='outside upper right')
			if out:
				fig.savefig(f"{out}{ff}w6_a_{ile}.png")
			else:
				plt.show()
			col[3]='r'
		else:
			print(f"plik zmiana_{ile} nie istnieje")
		###############################################################		
	fig,ax =plt.subplots(2,2,figsize=(20,20))
	axs = [ax[0,0],ax[0,1],ax[1,0],ax[1,1]]
	fig.supylabel("liczba błędnych przyporządkowań")
	fig.suptitle("Sekwencje przypisane do prawdziwej kategorii >= x")
	fig.supxlabel("x")
	ys=np.zeros((len(xs),4),dtype=int)
	for ai,a in enumerate(axs):
				
				
		for y in range(len(razem)):
			ys[y,:]=razem[y][ai]
		ylim=max(ys.sum(1))
		a.set_ylim(0,ylim)
		a.stackplot(xs,ys.T, labels=ki, alpha=0.8)
		a.legend(loc='upper right')
		a.set_title(f'prawdziwa kategoria: {ki[ai]}')
		a.set_xticks(xs)
	if out:
		fig.savefig(f"{out}{ff}w6_b.png")
	else:
		plt.show()
		
if args.m3:
	
	if os.path.isfile(met) and  os.path.isfile(ac):
		wczytaj_sl(ac,met)
		
		###
		ile =3
		###
		n1=f"{path}/stan_{ile}.txt"
		n1=n1.replace("//","/")
		if os.path.isfile(n1)		:		
			m5_1,m6_1=wczytaj_stan(n1)
			w1(m5_1,m6_1,7,[0,0,ile])
			if m5_1[1][1]!=[[],[]]:
				w1(m5_1,m6_1,7,[1,1,ile])
			if m5_1[2][2]!=[[],[]]:
				w1(m5_1,m6_1,7,[2,2,ile])
			if m5_1[3][3]!=[[],[]]:
				w1(m5_1,m6_1,7,[3,3,ile])
			w1(m5_1,m6_1,0,[0,0,ile])
			w1(m5_1,m6_1,0,[1,1,ile])
			w1(m5_1,m6_1,0,[2,2,ile])
			w1(m5_1,m6_1,0,[3,3,ile])
		else:
			print(f"brak pliku {n1}")
if args.m4:
	razem=[]
	xs=[]
	for ile in range(2,len(nazwy)+1):
		n=f"{path}/niepewny_{ile}.txt"
		n=n.replace("//","/")
		if os.path.isfile(n):				
			m5,m6=wczytaj_niepewny(n)
			ylim=max(m5.sum(1))
			xs.append(ile)
			razem.append(m5)
			######################################################
			fig,ax = plt.subplots(2,2,figsize=(20,20))
			axs=[ax[0,0],ax[0,1],ax[1,0],ax[1,1]]
			fig.suptitle(f"typy błędów\ndla sekwencji o kilku wzbudzonych neuronach >={ile} razy\nmodele: {nazwy}")
			fig.supxlabel("typ")
			fig.supylabel("liczba sekwencji")
			for ai,a in enumerate(axs):
				a.set_title(f"prawdziwa kategoria: {ki[ai]} czyli {ai}\nliczba błędów {sum(m6[ai,:])-m6[ai,ai]}, liczba poprawnych: {m6[ai,ai]}")
				col[ai]="orange"
				if ai:
					col[ai-1]="r"
				if not ai:
					aa=a.bar(range(16),m6[ai,:],color=col,label=lab)
				else:
					aa=a.bar(range(16),m6[ai,:],color=col)
				a.bar_label(aa)
			fig.legend(loc='outside upper right')
			if out:
				fig.savefig(f"{out}{ff}w8_2_{ile}.png")
			else:
				plt.show()
			col[3]='r'
			######################################################
			fig,ax =plt.subplots(figsize=(10,10))
			fig.supylabel("liczba błędów")
			fig.supxlabel("prawdziwa kategoria")
			ax.set_title(f"Sekwencje o kilku wzbudzonych neuronach >= {ile} razy z {len(nazwy)}")
			ax.set_ylim(0,ylim)
			bottom=[0,0,0,0]
					
			for r,d in enumerate(m5.T):
				a=ax.bar(ki, d,0.5, bottom=bottom, label=f"{ki[r]}")
				bottom=bottom+d
				ax.bar_label(a, label_type='center')		# label_type='center'

			fig.legend(title="przypisana kategoria")
			if out:
				fig.savefig(f"/{out}{ff}w8_0_{ile}.png")
			else:
				plt.show()
		else:
			print(f"plik zmiana_{ile} nie istnieje")
	###############################################################		
	fig,ax =plt.subplots(2,2,figsize=(20,20))
	axs = [ax[0,0],ax[0,1],ax[1,0],ax[1,1]]
	fig.supylabel("liczba błędnych przyporządkowań")
	fig.suptitle("Sekwencje o kilku neuronach wzbudzonych >= x razy")
	fig.supxlabel("x")
	ys=np.zeros((len(xs),4),dtype=int)
	for ai,a in enumerate(axs):	
		for y in range(len(razem)):
			ys[y,:]=razem[y][ai]
		ylim=max(ys.sum(1))
		a.set_ylim(0,ylim)
		a.stackplot(xs,ys.T, labels=ki, alpha=0.8)
		a.legend(loc='upper right')
		a.set_title(f'prawdziwa kategoria: {ki[ai]}')
		a.set_xticks(xs)
	if out:
		fig.savefig(f"{out}{ff}w8_1.png")
	else:
		plt.show()
		
if args.m5:
	met__ac = met_ac(path4)
	if met__ac:
		wczytaj_sl(met__ac[1],met__ac[0])
		for ile in  range(2, len(nazwy)+1):	# len(nazwy)+1 lub [3]
			print(f"ile = {ile}, liczba przedziałów = {args.bins}")
			n1=f"{path}/stan_{ile}.txt"
			n1=n1.replace("//","/")
			n2=f"{path}/zmiana_{ile}.txt"
			n2=n2.replace("//","/")
			if os.path.isfile(n1) and os.path.isfile(n2):
				m5_s,m6_s=wczytaj_stan(n1)
				m5_z,m6_z=wczytaj_45(n2)
				if isinstance(args.m5, list):
					w=list(dict.fromkeys(args.m5))
					for para in w:
						w1(m5_z,m5_s,9,[int(para[0]),int(para[2]),ile])
				else:
					w1(m5_z,m5_s,9,[int(args.m5[0]),int(args.m5[2]),ile])
			else:
				print(f"brakuje pliku {n1} lub {n2}")
