import re

def simple_tokenizer(text):
    """Simple tokenizer used for training and prediction.

    Keeps words of length >=2, lowercases and splits on non-word chars.
    """
    text = (text or '').lower()
    tokens = re.findall(r"\b\w\w+\b", text, flags=re.UNICODE)
    return tokens


french_stop_words = [
    'alors','au','aucuns','aussi','autre','avant','avec','avoir','bon','car','ce','cela',
    'ces','ceux','chaque','ci','comme','comment','dans','des','du','dedans','dehors','depuis',
    'devrait','doit','donc','dos','début','elle','elles','en','encore','essai','est','et','eu',
    'fait','faites','fois','font','hors','hélas','ici','il','ils','je','juste','la','le','les',
    'leur','là','ma','maintenant','mais','mes','mine','moins','mon','mot','même','ni','nommés',
    'notre','nous','nouveaux','ou','où','par','parce','parole','pas','personnes','peut','peu',
    'plupart','pour','pourquoi','quand','que','quel','quelle','quelles','quels','qui','sa','sans',
    'ses','seulement','si','sien','son','sont','sous','sur','ta','tandis','tellement','tels','tes',
    'ton','tous','tout','trop','très','tu','voient','vont','votre','vous','vu','ça','étaient','état',
    'étions','été','être'
]
