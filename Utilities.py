import pandas as pd
from changeo.Gene import getFamily, getGene
import numpy as np


alleles_mapping = {
    'IGHV1F1-G1*01': 'IGHV1-18*01',
    'IGHV1F1-G1*03': 'IGHV1-18*03',
    'IGHV1F1-G1*04': 'IGHV1-18*04',
    'IGHV1F1-G10*01': 'IGHV1-8*01',
    'IGHV1F1-G10*02': 'IGHV1-8*02',
    'IGHV1F1-G10*03': 'IGHV1-8*03',
    'IGHV1F1-G2*01': 'IGHV1-2*01',
    'IGHV1F1-G2*02': 'IGHV1-2*02',
    'IGHV1F1-G2*03': 'IGHV1-2*03',
    'IGHV1F1-G2*04': 'IGHV1-2*04',
    'IGHV1F1-G2*05': 'IGHV1-2*05',
    'IGHV1F1-G2*06': 'IGHV1-2*06',
    'IGHV1F1-G2*07': 'IGHV1-2*07',
    'IGHV1F1-G3*01': 'IGHV1-24*01',
    'IGHV1F1-G4*01': 'IGHV1-3*01',
    'IGHV1F1-G4*02': 'IGHV1-3*02',
    'IGHV1F1-G4*03': 'IGHV1-3*03',
    'IGHV1F1-G4*04': 'IGHV1-3*04',
    'IGHV1F1-G4*05': 'IGHV1-3*05',
    'IGHV1F1-G5*01': 'IGHV1-45*01',
    'IGHV1F1-G5*02': 'IGHV1-45*02',
    'IGHV1F1-G5*03': 'IGHV1-45*03',
    'IGHV1F1-G6*01': 'IGHV1-46*01',
    'IGHV1F1-G6*02': 'IGHV1-46*02',
    'IGHV1F1-G6*03': 'IGHV1-46*03',
    'IGHV1F1-G6*04': 'IGHV1-46*04',
    'IGHV1F1-G7*01': 'IGHV1-58*01',
    'IGHV1F1-G7*02': 'IGHV1-58*02',
    'IGHV1F1-G7*03': 'IGHV1-58*03',
    'IGHV1F1-G8*01': 'IGHV1-69-2*01',
    'IGHV1F1-G9*01': 'IGHV1-69D*01/IGHV1-69*01',
    'IGHV1F1-G9*01_C26T': 'IGHV1-69*01_C26T',
    'IGHV1F1-G9*02': 'IGHV1-69*02',
    'IGHV1F1-G9*03': 'IGHV1-69*04',
    'IGHV1F1-G9*03_T191C': 'IGHV1-69*04_T191C',
    'IGHV1F1-G9*04': 'IGHV1-69*05',
    'IGHV1F1-G9*05': 'IGHV1-69*06',
    'IGHV1F1-G9*05_G240A': 'IGHV1-69*06_G240A',
    'IGHV1F1-G9*06': 'IGHV1-69*08',
    'IGHV1F1-G9*07': 'IGHV1-69*09',
    'IGHV1F1-G9*08': 'IGHV1-69*10',
    'IGHV1F1-G9*08_A54G': 'IGHV1-69*10_A54G',
    'IGHV1F1-G9*09': 'IGHV1-69*11',
    'IGHV1F1-G9*10': 'IGHV1-69*12',
    'IGHV1F1-G9*11': 'IGHV1-69*13',
    'IGHV1F1-G9*12': 'IGHV1-69*14',
    'IGHV1F1-G9*13': 'IGHV1-69*15',
    'IGHV1F1-G9*14': 'IGHV1-69*16',
    'IGHV1F1-G9*15': 'IGHV1-69*17',
    'IGHV1F1-G9*16': 'IGHV1-69*18',
    'IGHV1F1-G9*17': 'IGHV1-69*19',
    'IGHV1F2-G11*01': 'IGHV2-26*01',
    'IGHV1F2-G11*02': 'IGHV2-26*02',
    'IGHV1F2-G11*03': 'IGHV2-26*03',
    'IGHV1F2-G11*04': 'IGHV2-26*04',
    'IGHV1F2-G12*01': 'IGHV2-5*01',
    'IGHV1F2-G12*02': 'IGHV2-5*02',
    'IGHV1F2-G12*05': 'IGHV2-5*05',
    'IGHV1F2-G12*06': 'IGHV2-5*06',
    'IGHV1F2-G12*08': 'IGHV2-5*08',
    'IGHV1F2-G12*09': 'IGHV2-5*09',
    'IGHV1F2-G13*01': 'IGHV2-70*01',
    'IGHV1F2-G13*02': 'IGHV2-70D*04/IGHV2-70*04',
    'IGHV1F2-G13*02_A14G': 'IGHV2-70*04_A14G',
    'IGHV1F2-G13*03': 'IGHV2-70*10',
    'IGHV1F2-G13*04': 'IGHV2-70*11',
    'IGHV1F2-G13*05': 'IGHV2-70*12',
    'IGHV1F2-G13*06': 'IGHV2-70*13',
    'IGHV1F2-G13*07': 'IGHV2-70*15',
    'IGHV1F2-G13*08': 'IGHV2-70*16',
    'IGHV1F2-G13*09': 'IGHV2-70*17',
    'IGHV1F2-G13*10': 'IGHV2-70*18',
    'IGHV1F2-G13*11': 'IGHV2-70*19',
    'IGHV1F2-G13*12': 'IGHV2-70*20',
    'IGHV1F2-G13*13': 'IGHV2-70D*14',
    'IGHV1F3-G14*01': 'IGHV3-11*01',
    'IGHV1F3-G14*03': 'IGHV3-11*03',
    'IGHV1F3-G14*04': 'IGHV3-11*04',
    'IGHV1F3-G14*05': 'IGHV3-11*05',
    'IGHV1F3-G14*06': 'IGHV3-11*06',
    'IGHV1F3-G15*01': 'IGHV3-13*01',
    'IGHV1F3-G15*01_G290A_T300C': 'IGHV3-13*01_G290A_T300C',
    'IGHV1F3-G15*02': 'IGHV3-13*02',
    'IGHV1F3-G15*03': 'IGHV3-13*03',
    'IGHV1F3-G15*04': 'IGHV3-13*04',
    'IGHV1F3-G15*05': 'IGHV3-13*05',
    'IGHV1F3-G17*01': 'IGHV3-20*01',
    'IGHV1F3-G17*04': 'IGHV3-20*04',
    'IGHV1F3-G18*01': 'IGHV3-21*01',
    'IGHV1F3-G18*02': 'IGHV3-21*02',
    'IGHV1F3-G18*03': 'IGHV3-21*03',
    'IGHV1F3-G18*04': 'IGHV3-21*04',
    'IGHV1F3-G18*05': 'IGHV3-21*05',
    'IGHV1F3-G18*06': 'IGHV3-21*06',
    'IGHV1F3-G19*01': 'IGHV3-23D*01/IGHV3-23*01',
    'IGHV1F3-G19*01_G239T': 'IGHV3-23*01_G239T',
    'IGHV1F3-G19*02': 'IGHV3-23*02',
    'IGHV1F3-G19*03': 'IGHV3-23*03',
    'IGHV1F3-G19*04': 'IGHV3-23*04',
    'IGHV1F3-G19*05': 'IGHV3-23*05',
    'IGHV1F3-G20*01': 'IGHV3-30-3*01',
    'IGHV1F3-G20*02': 'IGHV3-30-3*02',
    'IGHV1F3-G20*03': 'IGHV3-30-3*03/IGHV3-30*04',
    'IGHV1F3-G20*03_C201T_G317A': 'IGHV3-30*04_C201T_G317A',
    'IGHV1F3-G20*04': 'IGHV3-30-5*01/IGHV3-30*18',
    'IGHV1F3-G20*05': 'IGHV3-30-5*02/IGHV3-30*02',
    'IGHV1F3-G20*05_A275G': 'IGHV3-30*02_A275G',
    'IGHV1F3-G20*05_G49A': 'IGHV3-30*02_G49A',
    'IGHV1F3-G20*06': 'IGHV3-30*01',
    'IGHV1F3-G20*07': 'IGHV3-30*03',
    'IGHV1F3-G20*08': 'IGHV3-30*05',
    'IGHV1F3-G20*09': 'IGHV3-30*06',
    'IGHV1F3-G20*10': 'IGHV3-30*07',
    'IGHV1F3-G20*11': 'IGHV3-30*08',
    'IGHV1F3-G20*12': 'IGHV3-30*09',
    'IGHV1F3-G20*13': 'IGHV3-30*10',
    'IGHV1F3-G20*14': 'IGHV3-30*11',
    'IGHV1F3-G20*15': 'IGHV3-30*12',
    'IGHV1F3-G20*16': 'IGHV3-30*13',
    'IGHV1F3-G20*17': 'IGHV3-30*14',
    'IGHV1F3-G20*18': 'IGHV3-30*15',
    'IGHV1F3-G20*19': 'IGHV3-30*16',
    'IGHV1F3-G20*20': 'IGHV3-30*17',
    'IGHV1F3-G20*21': 'IGHV3-30*19',
    'IGHV1F3-G20*23': 'IGHV3-33*01',
    'IGHV1F3-G20*24': 'IGHV3-33*02',
    'IGHV1F3-G20*25': 'IGHV3-33*03',
    'IGHV1F3-G20*26': 'IGHV3-33*04',
    'IGHV1F3-G20*27': 'IGHV3-33*05',
    'IGHV1F3-G20*28': 'IGHV3-33*06',
    'IGHV1F3-G20*29': 'IGHV3-33*07',
    'IGHV1F3-G20*30': 'IGHV3-33*08',
    'IGHV1F3-G21*02': 'IGHV3-35*02',
    'IGHV1F3-G22*01': 'IGHV3-43*01',
    'IGHV1F3-G22*02': 'IGHV3-43*02',
    'IGHV1F3-G22*03': 'IGHV3-43D*03',
    'IGHV1F3-G22*04': 'IGHV3-43D*04',
    'IGHV1F3-G22*04_G4A': 'IGHV3-43D*04_G4A',
    'IGHV1F3-G23*01': 'IGHV3-48*01',
    'IGHV1F3-G23*02': 'IGHV3-48*02',
    'IGHV1F3-G23*03': 'IGHV3-48*03',
    'IGHV1F3-G23*04': 'IGHV3-48*04',
    'IGHV1F3-G25*01': 'IGHV3-53*01',
    'IGHV1F3-G25*02': 'IGHV3-53*02',
    'IGHV1F3-G25*02_C259T': 'IGHV3-53*02_C259T',
    'IGHV1F3-G25*03': 'IGHV3-53*03',
    'IGHV1F3-G25*04': 'IGHV3-53*04',
    'IGHV1F3-G25*05': 'IGHV3-53*05',
    'IGHV1F3-G25*06': 'IGHV3-66*01/IGHV3-66*04',
    'IGHV1F3-G25*07': 'IGHV3-66*02',
    'IGHV1F3-G25*07_G303A': 'IGHV3-66*02_G303A',
    'IGHV1F3-G25*08': 'IGHV3-66*03',
    'IGHV1F3-G26*01': 'IGHV3-64*01',
    'IGHV1F3-G26*02': 'IGHV3-64*02',
    'IGHV1F3-G26*03': 'IGHV3-64*03',
    'IGHV1F3-G26*04': 'IGHV3-64*04',
    'IGHV1F3-G26*05': 'IGHV3-64*05',
    'IGHV1F3-G26*06': 'IGHV3-64*07',
    'IGHV1F3-G26*07': 'IGHV3-64D*06',
    'IGHV1F3-G26*08': 'IGHV3-64D*08',
    'IGHV1F3-G26*09': 'IGHV3-64D*09',
    'IGHV1F3-G27*01': 'IGHV3-7*01',
    'IGHV1F3-G27*02': 'IGHV3-7*02',
    'IGHV1F3-G27*03': 'IGHV3-7*03',
    'IGHV1F3-G27*04': 'IGHV3-7*04',
    'IGHV1F3-G27*05': 'IGHV3-7*05',
    'IGHV1F3-G30*01': 'IGHV3-74*01',
    'IGHV1F3-G30*02': 'IGHV3-74*02',
    'IGHV1F3-G30*03': 'IGHV3-74*03',
    'IGHV1F3-G31*01': 'IGHV3-9*01',
    'IGHV1F3-G31*02': 'IGHV3-9*02',
    'IGHV1F3-G31*03': 'IGHV3-9*03',
    'IGHV1F4-G16*01': 'IGHV3-15*01',
    'IGHV1F4-G16*01_A313T': 'IGHV3-15*01_A313T',
    'IGHV1F4-G16*02': 'IGHV3-15*02',
    'IGHV1F4-G16*03': 'IGHV3-15*03',
    'IGHV1F4-G16*04': 'IGHV3-15*04',
    'IGHV1F4-G16*05': 'IGHV3-15*05',
    'IGHV1F4-G16*06': 'IGHV3-15*06',
    'IGHV1F4-G16*07': 'IGHV3-15*07',
    'IGHV1F4-G16*08': 'IGHV3-15*08',
    'IGHV1F4-G24*01': 'IGHV3-49*01',
    'IGHV1F4-G24*02': 'IGHV3-49*02',
    'IGHV1F4-G24*03': 'IGHV3-49*03',
    'IGHV1F4-G24*04': 'IGHV3-49*04',
    'IGHV1F4-G24*05': 'IGHV3-49*05',
    'IGHV1F4-G28*01': 'IGHV3-72*01',
    'IGHV1F4-G29*01': 'IGHV3-73*01',
    'IGHV1F4-G29*02': 'IGHV3-73*02',
    'IGHV1F5-G32*01': 'IGHV4-28*01',
    'IGHV1F5-G32*02': 'IGHV4-28*02',
    'IGHV1F5-G32*03': 'IGHV4-28*03',
    'IGHV1F5-G32*04': 'IGHV4-28*04',
    'IGHV1F5-G32*05': 'IGHV4-28*05',
    'IGHV1F5-G32*06': 'IGHV4-28*06',
    'IGHV1F5-G32*07': 'IGHV4-28*07',
    'IGHV1F5-G33*01': 'IGHV4-30-2*01',
    'IGHV1F5-G33*01_C285T': 'IGHV4-30-2*01_C285T',
    'IGHV1F5-G33*01_G70A': 'IGHV4-30-2*01_G70A',
    'IGHV1F5-G33*02': 'IGHV4-30-2*03',
    'IGHV1F5-G33*03': 'IGHV4-30-2*05',
    'IGHV1F5-G33*04': 'IGHV4-30-2*06',
    'IGHV1F5-G33*05': 'IGHV4-30-4*07',
    'IGHV1F5-G34*01': 'IGHV4-30-4*01',
    'IGHV1F5-G34*01_A70G_A107G': 'IGHV4-30-4*01_A70G_A107G',
    'IGHV1F5-G34*02': 'IGHV4-30-4*02',
    'IGHV1F5-G34*08': 'IGHV4-30-4*08',
    'IGHV1F5-G35*01': 'IGHV4-31*01',
    'IGHV1F5-G35*02': 'IGHV4-31*02',
    'IGHV1F5-G35*03': 'IGHV4-31*03',
    'IGHV1F5-G35*10': 'IGHV4-31*10',
    'IGHV1F5-G35*11': 'IGHV4-31*11',
    'IGHV1F5-G35*11_G4C_G21C_C25T_A113C': 'IGHV4-31*11_G4C_G21C_C25T_A113C',
    'IGHV1F5-G36*01': 'IGHV4-34*01',
    'IGHV1F5-G36*02': 'IGHV4-34*02',
    'IGHV1F5-G36*04': 'IGHV4-34*04',
    'IGHV1F5-G36*05': 'IGHV4-34*05',
    'IGHV1F5-G36*11': 'IGHV4-34*11',
    'IGHV1F5-G36*12': 'IGHV4-34*12',
    'IGHV1F5-G37*09': 'IGHV4-34*09',
    'IGHV1F5-G37*10': 'IGHV4-34*10',
    'IGHV1F5-G38*01': 'IGHV4-38-2*01',
    'IGHV1F5-G38*02': 'IGHV4-38-2*02',
    'IGHV1F5-G38*02_G246A': 'IGHV4-38-2*02_G246A',
    'IGHV1F5-G38*03': 'IGHV4-39*01',
    'IGHV1F5-G38*03_A200C': 'IGHV4-39*01_A200C',
    'IGHV1F5-G38*03_G315A': 'IGHV4-39*01_G315A',
    'IGHV1F5-G38*04': 'IGHV4-39*02',
    'IGHV1F5-G38*05': 'IGHV4-39*06',
    'IGHV1F5-G38*06': 'IGHV4-39*07',
    'IGHV1F5-G39*01': 'IGHV4-4*01',
    'IGHV1F5-G39*02': 'IGHV4-4*02',
    'IGHV1F5-G39*03': 'IGHV4-4*03',
    'IGHV1F5-G40*01': 'IGHV4-4*07',
    'IGHV1F5-G40*01_A70G': 'IGHV4-4*07_A70G',
    'IGHV1F5-G40*02': 'IGHV4-59*10',
    'IGHV1F5-G41*01': 'IGHV4-4*08',
    'IGHV1F5-G41*02': 'IGHV4-4*09',
    'IGHV1F5-G41*03': 'IGHV4-59*01',
    'IGHV1F5-G41*03_G267A': 'IGHV4-59*01_G267A',
    'IGHV1F5-G41*04': 'IGHV4-59*02',
    'IGHV1F5-G41*05': 'IGHV4-59*07',
    'IGHV1F5-G41*06': 'IGHV4-59*08',
    'IGHV1F5-G41*07': 'IGHV4-59*11',
    'IGHV1F5-G41*08': 'IGHV4-59*12',
    'IGHV1F5-G41*09': 'IGHV4-59*13',
    'IGHV1F5-G42*01': 'IGHV4-61*01',
    'IGHV1F5-G42*01_A41G': 'IGHV4-61*01_A41G',
    'IGHV1F5-G42*02': 'IGHV4-61*02',
    'IGHV1F5-G42*03': 'IGHV4-61*03',
    'IGHV1F5-G42*05': 'IGHV4-61*05',
    'IGHV1F5-G42*08': 'IGHV4-61*08',
    'IGHV1F5-G42*09': 'IGHV4-61*09',
    'IGHV1F5-G42*10': 'IGHV4-61*10',
    'IGHV1F6-G43*01': 'IGHV5-10-1*01',
    'IGHV1F6-G43*02': 'IGHV5-10-1*02',
    'IGHV1F6-G43*03': 'IGHV5-10-1*03',
    'IGHV1F6-G43*04': 'IGHV5-10-1*04',
    'IGHV1F6-G44*01': 'IGHV5-51*01',
    'IGHV1F6-G44*02': 'IGHV5-51*02',
    'IGHV1F6-G44*03': 'IGHV5-51*03',
    'IGHV1F6-G44*04': 'IGHV5-51*04',
    'IGHV1F6-G44*06': 'IGHV5-51*06',
    'IGHV1F6-G44*07': 'IGHV5-51*07',
    'IGHV1F7-G45*01': 'IGHV6-1*01',
    'IGHV1F7-G45*01_T91C': 'IGHV6-1*01_T91C',
    'IGHV1F7-G45*02': 'IGHV6-1*02',
    'IGHV1F8-G46*01': 'IGHV7-4-1*01',
    'IGHV1F8-G46*02': 'IGHV7-4-1*02',
    'IGHV1F8-G46*04': 'IGHV7-4-1*04',
    'IGHV1F8-G46*05': 'IGHV7-4-1*05',
    'IGHVF1-G1*01': 'IGHV3-72*01',
    'IGHVF1-G2*01': 'IGHV3-73*01',
    'IGHVF1-G2*02': 'IGHV3-73*02',
    'IGHVF1-G3*01': 'IGHV3-49*02',
    'IGHVF1-G3*02': 'IGHV3-49*01',
    'IGHVF1-G3*03': 'IGHV3-49*05',
    'IGHVF1-G3*04': 'IGHV3-49*03',
    'IGHVF1-G3*05': 'IGHV3-49*04',
    'IGHVF1-G4*01': 'IGHV3-15*07',
    'IGHVF1-G4*02': 'IGHV3-15*06',
    'IGHVF1-G4*03': 'IGHV3-15*05',
    'IGHVF1-G4*04': 'IGHV3-15*04',
    'IGHVF1-G4*05': 'IGHV3-15*02',
    'IGHVF1-G4*06': 'IGHV3-15*01',
    'IGHVF1-G4*07': 'IGHV3-15*01_A313T',
    'IGHVF1-G4*08': 'IGHV3-15*03',
    'IGHVF1-G4*09': 'IGHV3-15*08',
    'IGHVF2-G10*01': 'IGHV3-11*01',
    'IGHVF2-G10*02': 'IGHV3-11*04',
    'IGHVF2-G10*03': 'IGHV3-11*06',
    'IGHVF2-G10*04': 'IGHV3-11*03',
    'IGHVF2-G10*05': 'IGHV3-11*05',
    'IGHVF2-G11*01': 'IGHV3-21*07',
    'IGHVF2-G11*02': 'IGHV3-21*05',
    'IGHVF2-G11*03': 'IGHV3-21*06',
    'IGHVF2-G11*04': 'IGHV3-21*04',
    'IGHVF2-G11*05': 'IGHV3-21*03',
    'IGHVF2-G11*06': 'IGHV3-21*01',
    'IGHVF2-G11*07': 'IGHV3-21*02',
    'IGHVF2-G12*01': 'IGHV3-48*03',
    'IGHVF2-G12*02': 'IGHV3-48*04',
    'IGHVF2-G12*03': 'IGHV3-48*01',
    'IGHVF2-G12*04': 'IGHV3-48*02',
    'IGHVF2-G13*01': 'IGHV3-35*02',
    'IGHVF2-G14*01': 'IGHV3-30-3*02',
    'IGHVF2-G14*02': 'IGHV3-30*04_C201T_G317A',
    'IGHVF2-G14*03': 'IGHV3-30*08',
    'IGHVF2-G14*04': 'IGHV3-30*14',
    'IGHVF2-G14*05': 'IGHV3-30*09',
    'IGHVF2-G14*06': 'IGHV3-30-3*01',
    'IGHVF2-G14*07': 'IGHV3-30*04/IGHV3-30-3*03',
    'IGHVF2-G14*08': 'IGHV3-30*17',
    'IGHVF2-G14*09': 'IGHV3-30*16',
    'IGHVF2-G14*10': 'IGHV3-30*15',
    'IGHVF2-G14*11': 'IGHV3-30*11',
    'IGHVF2-G14*12': 'IGHV3-30*10',
    'IGHVF2-G14*13': 'IGHV3-30*01',
    'IGHVF2-G14*14': 'IGHV3-30*07',
    'IGHVF2-G14*15': 'IGHV3-30*20',
    'IGHVF2-G14*16': 'IGHV3-30*06',
    'IGHVF2-G14*17': 'IGHV3-30*19',
    'IGHVF2-G14*18': 'IGHV3-33*05',
    'IGHVF2-G14*19': 'IGHV3-30*03',
    'IGHVF2-G14*20': 'IGHV3-30*18/IGHV3-30-5*01',
    'IGHVF2-G14*21': 'IGHV3-30*12',
    'IGHVF2-G14*22': 'IGHV3-30*05',
    'IGHVF2-G14*23': 'IGHV3-30*13',
    'IGHVF2-G14*24': 'IGHV3-30*02_G49A',
    'IGHVF2-G14*25': 'IGHV3-30*02_A275G',
    'IGHVF2-G14*26': 'IGHV3-30*02/IGHV3-30-5*02',
    'IGHVF2-G14*27': 'IGHV3-33*02',
    'IGHVF2-G14*28': 'IGHV3-33*07',
    'IGHVF2-G14*29': 'IGHV3-33*04',
    'IGHVF2-G14*30': 'IGHV3-33*08',
    'IGHVF2-G14*31': 'IGHV3-33*03',
    'IGHVF2-G14*32': 'IGHV3-33*01',
    'IGHVF2-G14*33': 'IGHV3-33*06',
    'IGHVF2-G15*01': 'IGHV3-64*02',
    'IGHVF2-G15*02': 'IGHV3-64*01',
    'IGHVF2-G15*03': 'IGHV3-64*07',
    'IGHVF2-G15*04': 'IGHV3-64*04',
    'IGHVF2-G15*05': 'IGHV3-64D*06',
    'IGHVF2-G15*06': 'IGHV3-64D*08',
    'IGHVF2-G15*07': 'IGHV3-64D*09',
    'IGHVF2-G15*08': 'IGHV3-64*03',
    'IGHVF2-G15*09': 'IGHV3-64*05',
    'IGHVF2-G16*01': 'IGHV3-74*03',
    'IGHVF2-G16*02': 'IGHV3-74*01',
    'IGHVF2-G16*03': 'IGHV3-74*02',
    'IGHVF2-G17*01': 'IGHV3-66*04/IGHV3-66*01',
    'IGHVF2-G17*02': 'IGHV3-66*02',
    'IGHVF2-G17*03': 'IGHV3-66*02_G303A',
    'IGHVF2-G17*04': 'IGHV3-53*03',
    'IGHVF2-G17*05': 'IGHV3-53*05',
    'IGHVF2-G17*06': 'IGHV3-53*02_C259T',
    'IGHVF2-G17*07': 'IGHV3-53*01',
    'IGHVF2-G17*08': 'IGHV3-53*02',
    'IGHVF2-G17*09': 'IGHV3-53*04',
    'IGHVF2-G17*10': 'IGHV3-66*03',
    'IGHVF2-G18*01': 'IGHV3-23*02',
    'IGHVF2-G18*02': 'IGHV3-23*04',
    'IGHVF2-G18*03': 'IGHV3-23*01_G239T',
    'IGHVF2-G18*04': 'IGHV3-23D*01/IGHV3-23*01',
    'IGHVF2-G18*05': 'IGHV3-23*03',
    'IGHVF2-G18*06': 'IGHV3-23*05',
    'IGHVF2-G5*01': 'IGHV3-43D*04_G4A',
    'IGHVF2-G5*02': 'IGHV3-43D*03',
    'IGHVF2-G5*03': 'IGHV3-43D*04',
    'IGHVF2-G5*04': 'IGHV3-43*01',
    'IGHVF2-G5*05': 'IGHV3-43*02',
    'IGHVF2-G6*01': 'IGHV3-20*01',
    'IGHVF2-G6*02': 'IGHV3-20*04',
    'IGHVF2-G7*01': 'IGHV3-9*04',
    'IGHVF2-G7*02': 'IGHV3-9*03',
    'IGHVF2-G7*03': 'IGHV3-9*01',
    'IGHVF2-G7*04': 'IGHV3-9*02',
    'IGHVF2-G8*01': 'IGHV3-13*02',
    'IGHVF2-G8*02': 'IGHV3-13*03',
    'IGHVF2-G8*03': 'IGHV3-13*01_G290A_T300C',
    'IGHVF2-G8*04': 'IGHV3-13*05',
    'IGHVF2-G8*05': 'IGHV3-13*01',
    'IGHVF2-G8*06': 'IGHV3-13*04',
    'IGHVF2-G9*01': 'IGHV3-7*04',
    'IGHVF2-G9*02': 'IGHV3-7*01',
    'IGHVF2-G9*03': 'IGHV3-7*02',
    'IGHVF2-G9*04': 'IGHV3-7*03',
    'IGHVF2-G9*05': 'IGHV3-7*05',
    'IGHVF3-G19*01': 'IGHV5-10-1*02',
    'IGHVF3-G19*02': 'IGHV5-10-1*04',
    'IGHVF3-G19*03': 'IGHV5-10-1*01',
    'IGHVF3-G19*04': 'IGHV5-10-1*03',
    'IGHVF3-G20*01': 'IGHV5-51*02',
    'IGHVF3-G20*02': 'IGHV5-51*07',
    'IGHVF3-G20*03': 'IGHV5-51*06',
    'IGHVF3-G20*04': 'IGHV5-51*04',
    'IGHVF3-G20*05': 'IGHV5-51*01',
    'IGHVF3-G20*06': 'IGHV5-51*03',
    'IGHVF4-G21*01': 'IGHV7-4-1*01',
    'IGHVF4-G21*02': 'IGHV7-4-1*02',
    'IGHVF4-G21*03': 'IGHV7-4-1*04',
    'IGHVF4-G21*04': 'IGHV7-4-1*05',
    'IGHVF5-G22*01': 'IGHV1-24*01',
    'IGHVF5-G23*01': 'IGHV1-69-2*01',
    'IGHVF5-G24*01': 'IGHV1-58*03',
    'IGHVF5-G24*02': 'IGHV1-58*01',
    'IGHVF5-G24*03': 'IGHV1-58*02',
    'IGHVF5-G25*01': 'IGHV1-45*03',
    'IGHVF5-G25*02': 'IGHV1-45*01',
    'IGHVF5-G25*03': 'IGHV1-45*02',
    'IGHVF5-G26*01': 'IGHV1-69*02',
    'IGHVF5-G26*02': 'IGHV1-69*08',
    'IGHVF5-G26*03': 'IGHV1-69*10',
    'IGHVF5-G26*04': 'IGHV1-69*10_A54G',
    'IGHVF5-G26*05': 'IGHV1-69*20',
    'IGHVF5-G26*06': 'IGHV1-69*09',
    'IGHVF5-G26*07': 'IGHV1-69*04',
    'IGHVF5-G26*08': 'IGHV1-69*04_T191C',
    'IGHVF5-G26*09': 'IGHV1-69*17',
    'IGHVF5-G26*10': 'IGHV1-69*06',
    'IGHVF5-G26*11': 'IGHV1-69*06_G240A',
    'IGHVF5-G26*12': 'IGHV1-69*19',
    'IGHVF5-G26*13': 'IGHV1-69*18',
    'IGHVF5-G26*14': 'IGHV1-69*01_C26T',
    'IGHVF5-G26*15': 'IGHV1-69D*01/IGHV1-69*01',
    'IGHVF5-G26*16': 'IGHV1-69*14',
    'IGHVF5-G26*17': 'IGHV1-69*13',
    'IGHVF5-G26*18': 'IGHV1-69*05',
    'IGHVF5-G26*19': 'IGHV1-69*12',
    'IGHVF5-G26*20': 'IGHV1-69*16',
    'IGHVF5-G26*21': 'IGHV1-69*11',
    'IGHVF5-G26*22': 'IGHV1-69*15',
    'IGHVF5-G27*01': 'IGHV1-8*03',
    'IGHVF5-G27*02': 'IGHV1-8*01',
    'IGHVF5-G27*03': 'IGHV1-8*02',
    'IGHVF5-G28*01': 'IGHV1-46*04',
    'IGHVF5-G28*02': 'IGHV1-46*03',
    'IGHVF5-G28*03': 'IGHV1-46*01',
    'IGHVF5-G28*04': 'IGHV1-46*02',
    'IGHVF5-G29*01': 'IGHV1-2*07',
    'IGHVF5-G29*02': 'IGHV1-2*04',
    'IGHVF5-G29*03': 'IGHV1-2*02',
    'IGHVF5-G29*04': 'IGHV1-2*03',
    'IGHVF5-G29*05': 'IGHV1-2*01',
    'IGHVF5-G29*06': 'IGHV1-2*05',
    'IGHVF5-G29*07': 'IGHV1-2*06',
    'IGHVF5-G30*01': 'IGHV1-18*04',
    'IGHVF5-G30*02': 'IGHV1-18*01',
    'IGHVF5-G30*03': 'IGHV1-18*03',
    'IGHVF5-G31*01': 'IGHV1-3*05',
    'IGHVF5-G31*02': 'IGHV1-3*01',
    'IGHVF5-G31*03': 'IGHV1-3*04',
    'IGHVF5-G31*04': 'IGHV1-3*02',
    'IGHVF5-G31*05': 'IGHV1-3*03',
    'IGHVF6-G32*01': 'IGHV2-26*04',
    'IGHVF6-G32*02': 'IGHV2-26*03',
    'IGHVF6-G32*03': 'IGHV2-26*01',
    'IGHVF6-G32*04': 'IGHV2-26*02',
    'IGHVF6-G33*01': 'IGHV2-5*08',
    'IGHVF6-G33*02': 'IGHV2-5*01',
    'IGHVF6-G33*03': 'IGHV2-5*02',
    'IGHVF6-G33*04': 'IGHV2-5*09',
    'IGHVF6-G33*05': 'IGHV2-5*05',
    'IGHVF6-G33*06': 'IGHV2-5*06',
    'IGHVF6-G34*01': 'IGHV2-70*12',
    'IGHVF6-G34*02': 'IGHV2-70*10',
    'IGHVF6-G34*03': 'IGHV2-70D*14',
    'IGHVF6-G34*04': 'IGHV2-70*16',
    'IGHVF6-G34*05': 'IGHV2-70*17',
    'IGHVF6-G34*06': 'IGHV2-70*04_A14G',
    'IGHVF6-G34*07': 'IGHV2-70D*04/IGHV2-70*04',
    'IGHVF6-G34*08': 'IGHV2-70*01',
    'IGHVF6-G34*09': 'IGHV2-70*13',
    'IGHVF6-G34*10': 'IGHV2-70*11',
    'IGHVF6-G34*11': 'IGHV2-70*15',
    'IGHVF6-G34*12': 'IGHV2-70*18',
    'IGHVF6-G34*13': 'IGHV2-70*19',
    'IGHVF6-G34*14': 'IGHV2-70*20',
    'IGHVF7-G35*01': 'IGHV6-1*02',
    'IGHVF7-G35*02': 'IGHV6-1*01',
    'IGHVF7-G35*03': 'IGHV6-1*01_T91C',
    'IGHVF8-G36*01': 'IGHV4-4*08',
    'IGHVF8-G36*02': 'IGHV4-4*09',
    'IGHVF8-G36*03': 'IGHV4-59*08',
    'IGHVF8-G36*04': 'IGHV4-59*12',
    'IGHVF8-G36*05': 'IGHV4-59*13',
    'IGHVF8-G36*06': 'IGHV4-59*11',
    'IGHVF8-G36*07': 'IGHV4-59*07',
    'IGHVF8-G36*08': 'IGHV4-59*02',
    'IGHVF8-G36*09': 'IGHV4-59*01',
    'IGHVF8-G36*10': 'IGHV4-59*01_G267A',
    'IGHVF8-G37*01': 'IGHV4-59*10',
    'IGHVF8-G37*02': 'IGHV4-4*07',
    'IGHVF8-G37*03': 'IGHV4-4*07_A70G',
    'IGHVF8-G38*01': 'IGHV4-34*09',
    'IGHVF8-G38*02': 'IGHV4-34*10',
    'IGHVF8-G39*01': 'IGHV4-34*11',
    'IGHVF8-G39*02': 'IGHV4-34*12',
    'IGHVF8-G39*03': 'IGHV4-34*01',
    'IGHVF8-G39*04': 'IGHV4-34*02',
    'IGHVF8-G39*05': 'IGHV4-34*04',
    'IGHVF8-G39*06': 'IGHV4-34*05',
    'IGHVF8-G40*01': 'IGHV4-4*01',
    'IGHVF8-G40*02': 'IGHV4-4*03',
    'IGHVF8-G40*03': 'IGHV4-4*02',
    'IGHVF8-G40*04': 'IGHV4-4*10',
    'IGHVF8-G41*01': 'IGHV4-28*02',
    'IGHVF8-G41*02': 'IGHV4-28*06',
    'IGHVF8-G41*03': 'IGHV4-28*04',
    'IGHVF8-G41*04': 'IGHV4-28*07',
    'IGHVF8-G41*05': 'IGHV4-28*05',
    'IGHVF8-G41*06': 'IGHV4-28*01',
    'IGHVF8-G41*07': 'IGHV4-28*03',
    'IGHVF8-G42*01': 'IGHV4-30-2*03',
    'IGHVF8-G42*02': 'IGHV4-30-4*07',
    'IGHVF8-G42*03': 'IGHV4-30-2*05',
    'IGHVF8-G42*04': 'IGHV4-30-2*06',
    'IGHVF8-G42*05': 'IGHV4-30-2*01_G70A',
    'IGHVF8-G42*06': 'IGHV4-30-2*01',
    'IGHVF8-G42*07': 'IGHV4-30-2*01_C285T',
    'IGHVF8-G43*01': 'IGHV4-30-4*02',
    'IGHVF8-G43*02': 'IGHV4-30-4*01_A70G_A107G',
    'IGHVF8-G43*03': 'IGHV4-30-4*01',
    'IGHVF8-G43*04': 'IGHV4-30-4*08',
    'IGHVF8-G44*01': 'IGHV4-31*10',
    'IGHVF8-G44*02': 'IGHV4-31*11_G4C_G21C_C25T_A113C',
    'IGHVF8-G44*03': 'IGHV4-31*11',
    'IGHVF8-G44*04': 'IGHV4-31*02',
    'IGHVF8-G44*05': 'IGHV4-31*01',
    'IGHVF8-G44*06': 'IGHV4-31*03',
    'IGHVF8-G45*01': 'IGHV4-38-2*02_G246A',
    'IGHVF8-G45*02': 'IGHV4-38-2*01',
    'IGHVF8-G45*03': 'IGHV4-38-2*02',
    'IGHVF8-G45*04': 'IGHV4-39*08',
    'IGHVF8-G45*05': 'IGHV4-39*02',
    'IGHVF8-G45*06': 'IGHV4-39*01_G315A',
    'IGHVF8-G45*07': 'IGHV4-39*01',
    'IGHVF8-G45*08': 'IGHV4-39*01_A200C',
    'IGHVF8-G45*09': 'IGHV4-39*06',
    'IGHVF8-G45*10': 'IGHV4-39*07',
    'IGHVF8-G45*11': 'IGHV4-39*09',
    'IGHVF8-G46*01': 'IGHV4-61*09',
    'IGHVF8-G46*02': 'IGHV4-61*02',
    'IGHVF8-G46*03': 'IGHV4-61*11',
    'IGHVF8-G46*04': 'IGHV4-61*05',
    'IGHVF8-G46*05': 'IGHV4-61*10',
    'IGHVF8-G46*06': 'IGHV4-61*08',
    'IGHVF8-G46*07': 'IGHV4-61*03',
    'IGHVF8-G46*08': 'IGHV4-61*01',
    'IGHVF8-G46*09': 'IGHV4-61*01_A41G',
    'IGHVF2-G10*06': 'IGHV3-53*02_C259T',
    'IGHVF2-G10*07': 'IGHV3-53*02/IGHV3-53*01',
    'IGHVF2-G10*08': 'IGHV3-53*04',
    'IGHVF2-G10*09': 'IGHV3-53*05',
    'IGHVF2-G13*02': 'IGHV3-11*04',
    'IGHVF2-G13*03': 'IGHV3-11*06',
    'IGHVF2-G13*04': 'IGHV3-11*05/IGHV3-11*03',
    'IGHVF2-G16*04': 'IGHV3-7*03',
    'IGHVF2-G16*05': 'IGHV3-7*05',
    'IGHVF2-G17*11': 'IGHV3-30*01',
    'IGHVF2-G17*12': 'IGHV3-30*07',
    'IGHVF2-G17*13': 'IGHV3-30-3*02',
    'IGHVF2-G17*14': 'IGHV3-30*04_C201T_G317A',
    'IGHVF2-G17*15': 'IGHV3-30*06',
    'IGHVF2-G17*16': 'IGHV3-30*12',
    'IGHVF2-G17*17': 'IGHV3-30*05',
    'IGHVF2-G17*18': 'IGHV3-30*13',
    'IGHVF2-G17*19': 'IGHV3-30*19',
    'IGHVF2-G17*20': 'IGHV3-30*20',
    'IGHVF2-G17*21': 'IGHV3-33*05',
    'IGHVF2-G17*22': 'IGHV3-30*18/IGHV3-30-5*01',
    'IGHVF2-G17*23': 'IGHV3-30*03',
    'IGHVF2-G17*24': 'IGHV3-30*02_A275G',
    'IGHVF2-G17*25': 'IGHV3-30*02/IGHV3-30-5*02/IGHV3-30*02_G49A',
    'IGHVF2-G17*26': 'IGHV3-33*02',
    'IGHVF2-G17*27': 'IGHV3-33*07',
    'IGHVF2-G17*28': 'IGHV3-33*04',
    'IGHVF2-G17*29': 'IGHV3-33*08',
    'IGHVF2-G17*30': 'IGHV3-33*03',
    'IGHVF2-G17*31': 'IGHV3-33*01',
    'IGHVF2-G17*32': 'IGHV3-33*06',
    'IGHVF2-G9*06': 'IGHV3-64D*09',
    'IGHVF2-G9*07': 'IGHV3-64*03',
    'IGHVF2-G9*08': 'IGHV3-64*05',
    'IGHVF3-G18*01': 'IGHV5-10-1*02',
    'IGHVF3-G18*02': 'IGHV5-10-1*04',
    'IGHVF3-G18*03': 'IGHV5-10-1*03/IGHV5-10-1*01',
    'IGHVF3-G19*05': 'IGHV5-51*03/IGHV5-51*01',
    'IGHVF4-G20*01': 'IGHV7-4-1*01',
    'IGHVF4-G20*02': 'IGHV7-4-1*02',
    'IGHVF4-G20*03': 'IGHV7-4-1*04',
    'IGHVF4-G20*04': 'IGHV7-4-1*05',
    'IGHVF5-G21*01': 'IGHV1-24*01',
    'IGHVF5-G23*02': 'IGHV1-45*01',
    'IGHVF5-G23*03': 'IGHV1-45*02',
    'IGHVF5-G25*04': 'IGHV1-69*17',
    'IGHVF5-G25*05': 'IGHV1-69*10_A54G/IGHV1-69*10',
    'IGHVF5-G25*06': 'IGHV1-69*06_G240A',
    'IGHVF5-G25*07': 'IGHV1-69*14/IGHV1-69*06',
    'IGHVF5-G25*08': 'IGHV1-69*19',
    'IGHVF5-G25*09': 'IGHV1-69*05',
    'IGHVF5-G25*10': 'IGHV1-69*01_C26T/IGHV1-69*12/IGHV1-69*13/IGHV1-69D*01/IGHV1-69*01',
    'IGHVF5-G25*11': 'IGHV1-69*16',
    'IGHVF5-G25*12': 'IGHV1-69*02',
    'IGHVF5-G25*13': 'IGHV1-69*04/IGHV1-69*09',
    'IGHVF5-G25*14': 'IGHV1-69*04_T191C',
    'IGHVF5-G25*15': 'IGHV1-69*08',
    'IGHVF5-G27*04': 'IGHV1-2*01',
    'IGHVF5-G27*05': 'IGHV1-2*05',
    'IGHVF5-G27*06': 'IGHV1-2*06',
    'IGHVF6-G31*01': 'IGHV2-26*04',
    'IGHVF6-G31*02': 'IGHV2-26*03',
    'IGHVF6-G31*03': 'IGHV2-26*01',
    'IGHVF6-G31*04': 'IGHV2-26*02',
    'IGHVF6-G33*07': 'IGHV2-70*19',
    'IGHVF6-G33*08': 'IGHV2-70*20',
    'IGHVF6-G33*09': 'IGHV2-70*13',
    'IGHVF6-G33*10': 'IGHV2-70*01',
    'IGHVF6-G33*11': 'IGHV2-70*15/IGHV2-70*11',
    'IGHVF7-G34*01': 'IGHV6-1*01_T91C',
    'IGHVF7-G34*02': 'IGHV6-1*01/IGHV6-1*02',
    'IGHVF8-G35*01': 'IGHV4-34*11',
    'IGHVF8-G35*02': 'IGHV4-34*12',
    'IGHVF8-G35*03': 'IGHV4-34*02/IGHV4-34*01',
    'IGHVF8-G35*04': 'IGHV4-34*04',
    'IGHVF8-G35*05': 'IGHV4-34*05',
    'IGHVF8-G37*04': 'IGHV4-61*05',
    'IGHVF8-G37*05': 'IGHV4-61*10',
    'IGHVF8-G37*06': 'IGHV4-61*08',
    'IGHVF8-G37*07': 'IGHV4-61*03',
    'IGHVF8-G37*08': 'IGHV4-61*01_A41G/IGHV4-61*01',
    'IGHVF8-G38*03': 'IGHV4-30-4*01',
    'IGHVF8-G38*04': 'IGHV4-30-4*02',
    'IGHVF8-G38*05': 'IGHV4-31*10',
    'IGHVF8-G38*06': 'IGHV4-31*11',
    'IGHVF8-G38*07': 'IGHV4-31*11_G4C_G21C_C25T_A113C',
    'IGHVF8-G38*08': 'IGHV4-31*02',
    'IGHVF8-G38*09': 'IGHV4-31*01',
    'IGHVF8-G38*10': 'IGHV4-31*03',
    'IGHVF8-G39*07': 'IGHV4-30-2*01_C285T',
    'IGHVF8-G40*05': 'IGHV4-4*09',
    'IGHVF8-G40*06': 'IGHV4-59*08',
    'IGHVF8-G40*07': 'IGHV4-59*12',
    'IGHVF8-G40*08': 'IGHV4-59*13',
    'IGHVF8-G40*09': 'IGHV4-59*11',
    'IGHVF8-G40*10': 'IGHV4-59*02',
    'IGHVF8-G40*11': 'IGHV4-59*01_G267A',
    'IGHVF8-G40*12': 'IGHV4-59*01/IGHV4-59*07',
    'IGHVF8-G41*08': 'IGHV4-39*09',
    'IGHVF8-G41*09': 'IGHV4-39*06',
    'IGHVF8-G41*10': 'IGHV4-39*07'
}


def get_imgt_allele(functional_group_allele):
    """
    maps a functional_group_allele to imgt allele, if there is no mapping the functional_group_allele is returned
    :param functional_group_allele: a functional_group_allele
    :return: the imgt allele that matches the functional_group_allele
    """
    if functional_group_allele in alleles_mapping:
        return alleles_mapping[functional_group_allele]
    elif functional_group_allele.split('_')[0] in alleles_mapping:
        return alleles_mapping[functional_group_allele.split('_')[0]]
    return functional_group_allele


def build_feature_table(df: pd.DataFrame, mode: str = 'binary') -> pd.DataFrame:
    """
    builds a feature table where rows are the repertoire samples and the columns are the clusters
    :param df: airr-seq data frame
    :param mode: how to compute the values in the feature table, options are normalized, freq or binary
    :return: a feature table
    """
    # create feature table of subjects vs clusters frequency
    if mode == 'normalized':
        # cluster proportion in repertoire
        feature_table = df.groupby(['study_id', 'subject_id']).apply(
            lambda x: pd.DataFrame(
                x.groupby('cluster_id').apply(lambda y: y.duplicate_count.sum()) / x.duplicate_count.sum()
            ).transpose().reset_index(drop=True)
        ).droplevel(2).fillna(0)
    elif mode == 'freq':
        # cluster occurrences in repertoire
        feature_table = df.groupby(['study_id', 'subject_id']).apply(
            lambda x: pd.DataFrame(
                x.cluster_id.value_counts()
            ).transpose().reset_index(drop=True)
        ).droplevel(2).fillna(0)
    elif mode == 'binary':
        # cluster existence in repertoire
        feature_table = df.groupby(['study_id', 'subject_id']).apply(
            lambda x: pd.DataFrame(np.ones(len(x.cluster_id.unique())), index=x.cluster_id.unique()).transpose()
        ).droplevel(2).fillna(0)
    else:
        assert False, 'unsupported mode value'

    return feature_table


def load_sampled_airr_seq_df(
    file_path: str,
    labels: pd.Series = None,
    v_call_field: str = "v_call_original",
    group_mode: str = "family"
) -> pd.DataFrame:
    """
    load sequence_df and set index and important fields, if labels provided filter the sequence_df according
    :param file_path: the full file path of the sampled airr-seq data frame tsv file
    :param labels: labels of samples to filter the sequence_df data frame by
    :param v_call_field: which column in the dataframe use for the v_group
    :param group_mode: weather to populate the v_group field with the v_family or the v_gene
    :return: the loaded data frame filtered by labels, with v_group and j_group fields added and unique id index
    """
    airr_seq_df = pd.read_csv(
        file_path,
        dtype={'study_id': str, 'subject_id': str, 'sequence_id': str},
        sep='\t'
    )
    if labels is not None:
        # filter only sequences that belongs to samples in the labels series
        airr_seq_df = airr_seq_df.set_index(['study_id', 'subject_id']).loc[labels.index].reset_index()
    # adjust the v_call_original field - relevant if the v_call field contains functional group assignment
    airr_seq_df['v_call_original'] = airr_seq_df.v_call.apply(
        lambda x: ','.join(list(map(lambda y: get_imgt_allele(y).replace('/', ','), x.split(','))))
    )
    airr_seq_df['v_group'] = airr_seq_df[v_call_field].apply(getFamily) if group_mode == "family" else airr_seq_df[v_call_field].apply(getFamily)
    airr_seq_df['j_group'] = airr_seq_df.j_call.apply(getGene)
    airr_seq_df['v_gene'] = airr_seq_df[v_call_field].apply(getGene)
    airr_seq_df['junction_aa_length'] = airr_seq_df.junction_aa.str.len().astype(str)
    airr_seq_df['id'] = airr_seq_df.study_id + ';' + airr_seq_df.subject_id + ';' + airr_seq_df.sequence_id
    return airr_seq_df.set_index('id')


def load_metadata(file_path: str) -> pd.DataFrame:
    """
    load an airr-rep metadata tsv file
    :param file_path:
    :return: the loaded data frame
    """
    return pd.read_csv(
        file_path, sep='\t', dtype={'subject_id': 'str', 'study_id': 'str'}
    ).set_index(['study_id', 'subject_id'])


def filter_airr_seq_df_by_labels(airr_seq_df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """
    filter airr-seq data frame to sequences belonging to samples in labels
    :param airr_seq_df: airr-seq data frame
    :param labels: series of labels
    :return: filtered airr-seq data frame
    """
    return airr_seq_df.reset_index().set_index(['study_id', 'subject_id'], drop=False).loc[labels.index].set_index('id')