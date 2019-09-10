import re

mispell_dict_pt = {'c/':'com',
                's/':'sem',
                'p/':'para',
                '1/2':'meio',
                '1/4':'um quarto',
                'pçs':'peças',
                'hidrovacuo':'hidrovácuo',
                'homocinetica':'homocinética',
                'reservatorio':'reservatório',
                'texx':'motociclista',
                'xxcm':'',
                'suspensao':'suspensão',
                'taiff':'counseling',
                'cooktop':'cozinhar topo',
                'bieleta':'barra de oscilação',
                'bauleto':'caixa de motocicleta',
                'yongnuo':'câmera flash',
                'recarregavel':'recarregável',
                'cmxcm':'',
                'potenciometro': 'potenciômetro',
                'ombrelone': 'guarda sol',
                'burigotto': 'produtos bebê',
                'parabarro':'carro fender',
                'jolitex': 'cobertor',
                'impermeavel':'impermeável',
                'espiao':'espião',
                'acrigel':'pedicure',
                'ignicao':'ignição',
                'dobravel':'dobrável',
                'galzerano':'produtos bebê',
                'espatula':'espátula',
                'xxmm': '',
                'pineng': 'carregador portátil de celular',
                'multifilamento': 'vários filamentos',
                'fogao':'fogão',
                'guidao': 'guidão',
                'monopods' : 'mono pé'
                }

mispell_dict_es = {'&':'y',
                'n°':'',
                'bujia':'encendido',
                'bestway':'piscinas inflables',
                'monocomando':'grifos',
                '°':'',
                'homocinetica':'homocinética',
                '®':'',
                'bieleta':'barra oscilante',
                'vanitory':'baños de baño',
                'practicuna':'playards de bebé',
                'bujias':'encendido',
                'xxcm':'',
                'cooktop':'cocinar encima',
                'intex':'piscinas inflables',
                'crapodina':'rodamiento de embrague',
                'calefon':'calentadores de agua',
                'guardaplast':'revestimientos de guardabarros',
                'multifuncion':'muchas funciones',
                'muresco': 'papel pintado',
                'cotillon': 'cotillón',
                'electrogeno': 'generadores portátiles',
                'chatitas':'pisos',
                'jgo' : 'juego',
                'bordeadora':'recortadoras de cuerda',
                'bulon':'bulón',
                'fibrofacil':'fibra de madera',
                'pietcard':'módulos motocicleta',
                'desmalezadora':'cortadores de ceppilos',
                'banderin':'banderín',
                'monopatin':'scooter',
                'nutrilon': 'fórmula para bebés',
                'velocimetro': 'velocímetro'
                }


def clean_numbers(x):

    x = re.sub('[0-9]', '', x)
    return x

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


def replace_typical_misspell(text, language):
    if language == 'portuguese':
        mispellings, mispellings_re = _get_mispell(mispell_dict_pt)
    else:
        mispellings, mispellings_re = _get_mispell(mispell_dict_es)


    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)

def clean_text(x):

    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x
