""" this library is just a funny memory, look at guezzi.py for
the new library.
"""
import math
import pyperclip
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import re


#esegue la media semplice
def mean(lista : list[float]) -> float:
    """
    Calcola la media aritmetica di una lista di numeri.

    Args:
        lst (list[float]): Lista di numeri.

    Returns:
        float: Media aritmetica della lista.
    """
    
    return sum(lista)/len(lista)

#utile nelle altre funzioni
def zip_product(list1: list[float], list2: list[float]) -> list[float]:
    """
    Restituisce una lista in cui l'i-esimo elemento è il prodotto degli i-esimi elementi delle due liste date come argomenti.

    Args:
        list1 (list[float]): Prima lista di numeri.
        list2 (list[float]): Seconda lista di numeri.

    Returns:
        list[float]: Lista dei prodotti degli elementi corrispondenti delle due liste.
    """
    if len(list1) == len(list2):
        return [list1[i] * list2[i] for i in range(len(list1))]

#esegue la media pesata
def w_mean(values_with_errors: dict[float, float]) -> dict[float, float]:
    """
    Calcola la media pesata dei valori con errori associati.

    Args:
        values_with_errors (dict[float, float]): Dizionario con keys i valori e values gli errori associati.

    Returns:
        dict[float, float]: Dizionario contenente la media pesata e il suo errore.
    """ 
    #riceve un dizionario con keys i valori e values gli errori
    values = list(values_with_errors.keys()) #chiedo venia per la confusione
    weights = list(values_with_errors.values())
    weight_list = []
    for i in weights:
        weight_list.append(1/(i**2)) #secondo la formula
    weighted_values = zip_product(values, weight_list)
    w_mean = sum(weighted_values)/sum(weight_list)  #la media pesata
    error = 1/math.sqrt(sum(weight_list))  #l'errore
    return {w_mean: error} #mi restituisce una dizionario con due soli valori 

#utile nelle altre funzioni
def lists_to_dict(keys_list, values_list):
    #fa qualcosa solo se le liste sono della stessa lunghezza
    if len(keys_list) == len(values_list):
        result_dict = dict(zip(keys_list, values_list))
        return result_dict

#calcolo della deviazione standard    
def std_dev(list):
    varianza = 0
    for i in list:
        varianza += ((i-mean(list))**2)/(len(list)-1)
    return math.sqrt(varianza)

#errorer standard della media
def std_err(list):
    return std_dev(list)/math.sqrt(len(list))

#minimi quadrati, semplici, pesati, con errori omogenei su x e y (dy equivalente)
def lst_squares(x, y): 
    if len(x)!=len(y):
        print("deve esserci lo stesso numero di x e y misurate")
        return 1
    if type(x) is dict and type(y) is list:
        print("risultati con x e y scambiati")
        tmp = x
        x = y
        y = tmp

#PREAMBOLO
    minimi_semplici = False

    if type(y) is list:
        x_list = x
        y_list = y
        dy =  0 #errore sulle y provvisorio
        minimi_semplici = True
        delta_ = (len(x_list)*sum(x_squares))-(sum(x_list)**2)

    #valori associati al proprio errore in un dizionario    
    if type(y) is dict:
        y_list = list(y.keys())
        y_errors = list(y.values())
        weights = [1/(i**2) for i in y_errors]

        #controllo minimi quadrati semplici o pesati
        if all(y_err == y_errors[0] for y_err in y_errors):
            if type(x) is dict:
                x_errors = list(x.values())
                x_list = list(x.keys())
                dy = y_errors[0]
                if not all(x_err == x_errors[0] for x_err in x_errors):
                    print("tutti i dy uguali e i dx uguali se errori sia su x che su y")
                    return 1
            
            if type(x) is list:
                x_list = x
                dy = y_errors[0]

            x_squares = [x**2 for x in x_list]
            minimi_semplici = True
            delta_ = (len(x_list)*sum(x_squares))-(sum(x_list)**2)

        else:
            if type(x) is dict:
                print("tutti i dy uguali e i dx uguali se errori sia su x che su y")
                return 1
            else:
                x_list = x
                x_squares = [x**2 for x in x_list]
            delta_ = (sum(weights)*sum(zip_product(x_squares, weights)))-(sum(zip_product(weights, x_list)))**2
    
#######################################################################################################

    if not minimi_semplici:
        #minimi quadrati pesati
        a_numerator = (sum(zip_product(weights, x_squares))*sum(zip_product(weights, y_list)))- \
        (sum(zip_product(weights, x_list))*sum(zip_product(weights, zip_product(x_list, y_list))))
        
        a = a_numerator/delta_
        
        b_numerator = (sum(weights)*sum(zip_product(weights, zip_product(x_list, y_list))))- \
            (sum(zip_product(weights, x_list))*sum(zip_product(weights, y_list)))
        b = b_numerator/delta_

        #non abbiamo ancora gli strumenti per trovare gli errori su A e B usando diversi errori
        #oppure considerando sia gli errori sulle x sia sulle y
        da = math.sqrt(sum(zip_product(weights, x_squares))/delta_)
        db = math.sqrt(sum(weights)/delta_)
####################################################################################################
        
    if minimi_semplici:
        #minimi quadrati normali
        if len(x_list)==len(y_list):
            a_numerator = ((sum(x_squares)*sum(y_list))-(sum(zip_product(x_list, y_list))*sum(x_list)))
            delta_ = (len(x_list)*sum(x_squares))-(sum(x_list)**2)
            a = a_numerator/delta_

            b_numerator = (len(x_list)*sum(zip_product(x_list, y_list)))-(sum(y_list)*sum(x_list))
            delta_ = delta_
            b = b_numerator/delta_

            #ipotesi distribuzione normale!
            if dy == 0:
                dy = sum((y_list[i]-a-(b*x_list[i]))**2 for i in range(len(x_list)))/(len(x_list)-2)
            
            if type(x) is dict:
                #dy equivalente
                dy = math.sqrt(dy**2 + (b * x_errors[0])**2)
                y_errors = [dy for i in range(len(y_errors))]
                
            #formula errori su A e B       
            da = math.sqrt(sum(x_squares)/delta_) * dy
            db = math.sqrt(len(x_list)/delta_) * dy
            
    return [a, b, da, db], y_list, y_errors

#restituisce chi ridotto, copia un codice latex con tutti dati, printa la probabilità su terminale
def chi_quadro(x_list, y, copy = True):
    list_, y_list, y_errors = lst_squares(x_list, y)
    a = list_[0]
    b = list_[1]
    chi_quadro =sum(((y_list[i]-a-(b*x_list[i]))/y_errors[i])**2 for i in range(len(x_list)))
    gradi_liberta = len(x_list) - 2
    chi_quadro_ridotto = chi_quadro/gradi_liberta

    # Funzione di densità di probabilità
    x, k = sp.symbols('x k')
    f_k = (2 / (2**(k/2) * sp.gamma(k/2))) * x**(k - 1) * sp.exp(-x**2 /2)

    integral_result = sp.integrate(f_k, (x, (math.sqrt(chi_quadro), sp.oo)))
    prob_x2_maggiore_x02 = integral_result.subs(k, gradi_liberta+0.000000001).evalf()
    percentuale = prob_x2_maggiore_x02 * 100
    if percentuale > 5:
        sign = ">"
    else:
        sign = "<"

    #piccola chicca
    if chi_quadro_ridotto > 200:
        print("piacere, sono il brasiliano")

    print("probabilità chi quadro =", prob_x2_maggiore_x02)
    if (copy == True):
        latex_chi = (rf"$\tilde{{\chi_{{0}}}}^2 = {chi_quadro_ridotto:.3f}$ con {gradi_liberta} gradi di libertà "
            rf"\[P(\tilde{{\chi}}^2 \geq \tilde{{\chi_{{0}}}}^2) \approx {percentuale:.2f}\% {sign} 5\% \]")
        pyperclip.copy(latex_chi)
    return chi_quadro_ridotto
        
#tabella latex da liste python o testo csv ()
def latex_table(*args):
    """"""
    latex_code1=""
    if len(args) == 1:
        input = args[0]
    elif len(args) > 1:
        input = list(args)
    #accetta testo in formato csv, non accetta direttamente dei file
    if isinstance(input, str):
        rows = input.strip().split('\n')
        for row in rows:
        #separa gli elementi di ogni riga
            cells = row.split(',')
            row_length = len(cells)
            latex_code1 += " & ".join(cells) + " \\\\ \\hline\n"
    
    #accetta lista di liste di valori o una lista di valori
    elif isinstance(input, list):
        #ogni lista occupa una riga
        try:
            row_length = len(input[0])
            for row in input:
                latex_code1 += " & ".join(map(str, row)) + " \\\\ \\hline\n"
        #una lista di valori viene messa in una riga
        except TypeError:
            row_length = len(input)
            latex_code1 += " & ".join(map(str, input)) + " \\\\ \\hline\n"

        
    
    #codice latex per la tabella
    latex_code = "\\begin{center}\n"
    latex_code += "\\boxed{\n"
    latex_code += "\\begin{tabular}{|" + "|".join(["c"] * row_length) + "|}\n"
    latex_code += "\\hline\n"
    latex_code += latex_code1 
    latex_code += "\\end{tabular}%\n"
    latex_code += "}\n"
    latex_code += "\\captionof{table}{captionoftable here}\n"
    latex_code += "\\label{tab:your_table_label}\n"
    latex_code += "\\end{center}"
    
    #la tabella viene copiata in automatico
    pyperclip.copy(latex_code)
    
    #nel caso boh
    return latex_code


def create_best_fit_line(*args):
    latex_grafico = "\\begin{center}\n  \\includegraphics[width=0.8\\textwidth]\n  {grafici/}\n\\end{center}"
    pyperclip.copy(latex_grafico)
    #accetta liste di valori, ogni coppia deve avere la stessa lunghezza, se metti piu coppie
    #verranno printate sullo stesso grafico
    if len(args)%2!=0:
        return False
    x_label = input("qual è il nome dell'asse x? ")
    y_label = input("qual è il nome dell'asse y? ")
    titolo = input("qual è il titolo del grafico? ")
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(titolo)

    #reitera sulle coppie di liste negli argomenti
    for i in range(0, len(args), 2):
        list1 = args[i]
        list2 = args[i+1]
        if type(list1) is dict:
            list1 = list(list1.keys())
        if type(list2) is dict:
            list2 = list(list2.keys())

        #controllo match lunghezze
        if len(list1) == len(list2):
            x = np.array(list1)
            y = np.array(list2)
            #assicura che sia smooth
            x_values = np.linspace(min(x), max(x), 100)
            #personalizzazione di ogni elemento
            colore = input("qual è il colore della linea? ")
            nome_retta = input("qual è il nome della linea? ")
            #tipo di interpolazione
            grado = int(input("qual è il grado? "))
            
            coeffs = np.polyfit(x, y, grado)
            print(coeffs)

            
            plt.scatter(list1, list2)
            plt.plot(x_values, np.polyval(coeffs, x_values), color=colore, label= nome_retta)
            plt.legend()
            
    plt.show()

#inutile
"""def err_prop_input(num_fattori, coeff):
    num_fattori = int(num_fattori)
    coeff = float(coeff)
    f_best = 1.0*coeff
    formula = 0.0
    for _ in range(num_fattori):
        val, err, esp = map(float, input("valore,errore,esponente della misura: ").split(","))
        f_best *= val**esp
        formula += (esp*err/val)**2
    f_err = (math.sqrt(formula))*f_best
    return f_best, f_err"""

#usare eprop, mi serve per i file vecchi
def err_prop(list_of_lists, coeff):
    coeff = float(coeff)
    f_best = 1.0*coeff
    formula = 0.0
    for i in list_of_lists:
        val, err, esp = map(float, i)
        f_best *= val**esp
        formula += (esp*err/val)**2
    f_err = (math.sqrt(formula))*f_best
    return f_best, f_err

#restituisce il valore di 1 - P(entro t sigma)
def test_comp(a, sigma_a, b, sigma_b):

    #si suppongono le variabili indipendenti
    sigma_eq = math.sqrt(sigma_a**2 + sigma_b**2)
    t_sigma =  abs(a-b)/sigma_eq
    t = sp.symbols('t')
    prob_entro_t_sigma = sp.erf(t / sp.sqrt(2)).subs(t, t_sigma).evalf()
    if prob_entro_t_sigma > 0.95:
        print("Ci che hai fatto")
    print("probabilità entro t sigma =", prob_entro_t_sigma)
    return 1 - prob_entro_t_sigma
    

def eprop(formula: str, *args, copy = True):
    """calcolo propagazione errori su qualsiasi formula, bisogna inserire in una lista o
    come argomenti i valori e gli errori (in ordine come compaiono nella formula)
    per funzioni complesse usare sp.<nome funzione> tipo sp.sin() sp.sqrt() ecc.
    nel caso consultare la documentazione di sympy

    esempio: eprop("v = sp.sqrt( F / (2 * m) * t**2 )", [0.45, 0.03, 0.30, 0.01, 5.0, 0.1])
    con F=0.45, dF=0.03, m=0.30, dm=0.01, t=5.0, dt=0.1

    restituisce v e sigma v, inoltre copia un codice latex con formule e risultati di entrambe

    1) ogni variabile nella formula deve essere rappresentata con una lettera, ogni lettera "isolata"
    è considerata come variabile con errore, la prima lettera deve essere la variabile dipendente,
    e bisogna rispettare la sintassi di python nel definirla

    non valido:
        f = 0.5x + 1, 2*f = x + 2
    valido:
        f = 0.5 * x + 1, f = 1/2 * (x + 2), f = 1/2 * x + 1

    2) vengono create/modificate variabili di tipo sp.symbol con le lettere, quindi se si fa
    eprop("v = sp.sqrt( F / (2 * m) * t**2 )", [F, sigma_F, m, sigma_m, t, sigma_t])
    il programma non funziona, si possono ovviamente inserire i nomi di dati gia trovati ma, in generale,
    TUTTE LE LETTERE NELLE FORMULE NON DOVREBBERO CORRISPONDERE AI NOMI DI DATI GIA ESISTENTI
    quindi potrò per esempio fare
    eprop("v = sp.sqrt( F / (2 * m) * t**2 )", F_val, dF, m_, sigma_m_, T, sigma_T)
    se tali valori sono gia stati definiti)"""
    if len(args) % 2 != 0 and len(args)!=1 :
        print("uso incorretto")
        return False
    valori = []
    errori = []
    if len(args)==1:
        dati = args[0]
    else:
        dati = args

    for i in range(0, len(dati), 2):
        valori.append(dati[i])
        errori.append(dati[i+1])

    #tratta come variabili della funzione f tutte le lettere non adiacenti ad altre lettere
    pattern = r'\b([a-zA-Z])\b'
    variabili_str = re.findall(pattern, formula)

    f_variable = variabili_str[0]

    #crea variabili globali con il nome delle lettere e degli errori e le rende utilizzabili con sympy
    for i in range(1, len(variabili_str)):
        var_symbol = sp.symbols(variabili_str[i])
        sigma_symbol = sp.symbols(f"sigma_{variabili_str[i]}")
        
        globals()[variabili_str[i]] = var_symbol
        globals()[f"sigma_{variabili_str[i]}"] = sigma_symbol

    local_vars = {}
    exec(formula, globals(), local_vars)
    f = local_vars[f_variable]
    globals()[f_variable] = f
    
    #creazione derivate parziali
    for i in range(1, len(variabili_str)):
        globals()[f"partial_{variabili_str[i]}"] = sp.diff(globals()[variabili_str[0]], globals()[variabili_str[i]])
    
    #equazione per l'errore
    globals()[f"sigma_{variabili_str[0]}_squared"] = 0
    for i in range(1, len(variabili_str)):
        globals()[f"sigma_{variabili_str[0]}_squared"] += (globals()[f"partial_{variabili_str[i]}"] * globals()[f"sigma_{variabili_str[i]}"])**2
    globals()[f"sigma_{variabili_str[0]}"] = sp.sqrt(globals()[f"sigma_{variabili_str[0]}_squared"])

    sostituzioni = {}
    for i in range(1, len(variabili_str)):
        sostituzioni[globals()[variabili_str[i]]] = valori[i-1]
        sostituzioni[globals()[f"sigma_{variabili_str[i]}"]] = errori[i-1]

    risultato_errore = globals()[f"sigma_{variabili_str[0]}"].subs(sostituzioni).evalf()
    risultato_valore = globals()[variabili_str[0]].subs(sostituzioni).evalf()

    #copia formule e risultato in codice latex
    if copy == True:
        testo_latex = (
            rf"\[{variabili_str[0]} = {sp.latex(globals()[f_variable])} = {risultato_valore}\] \\ "
            rf"\[\sigma_{variabili_str[0]} = {sp.latex(globals()[f'sigma_{f_variable}'])} = {risultato_errore}\]"
        )
        pyperclip.copy(testo_latex)
    return risultato_valore, risultato_errore


    