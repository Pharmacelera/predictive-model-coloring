#!/usr/bin/env python

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import rdqueries
from rdkit.Chem import rdDepictor
from rdkit.Chem import FragmentCatalog
from rdkit import RDConfig
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.Draw import SimilarityMaps
from pathlib import Path
import re

#
# Copyright (c) 2021 Pharmacelera S.L.
# All rights reserved.
#
# Description: Script for getting atom or fragment list.
#
# Usage: Define the SMILES and interpretation level and run this script
#

class Get_Fragment_lst():
    """Script for getting atom or fragment list.

    Parameters
    ----------
    smiles : a single smiles for interpretation.
    level : level of interpretation. atom: Atom level. frag: fragment level defeind by Get_Fragment_lst method. Default is 'frag'.
    """
    def __init__(self,smiles,level='frag'):
        self.smiles=smiles
        if level=='atom':            
            self.mol_list=self.get_atom_lst()
        elif level=='frag':
            self.mol_list=self.get_fragment()
        else:
            self.mol_list=[]
            print('Please set level parameter with atom or frag')
    def draw_mol(self):
        m2=Chem.MolFromSmiles(self.smiles)
        for atom in m2.GetAtoms():
            atom.SetProp('atomLabel',str(atom.GetIdx()))
        return m2
    def get_atom_lst(self):
        lst=[]
        mol=Chem.MolFromSmiles(self.smiles)
        for i in range(mol.GetNumAtoms()):
            lst.append([i])
        return lst
    def rm_small_frag(self,atom_lst):
        atom_lst2=[]
        for m in range(len(atom_lst)):
            t1=set(atom_lst[m])
            subset=False
            for j in range(len(atom_lst)):
                t2=set(atom_lst[j])
                if t1<t2:
                    subset=True
                    break
            if not subset:
                atom_lst2.append(list(t1))
        return atom_lst2
    def rm_small_frag2(self,atom_lst):
        atom_lst3 = sorted(atom_lst,key = lambda i:(len(i),i),reverse=True) 
        atom_lst2=[]
        for m in range(len(atom_lst3)):
            t1=set(atom_lst3[m])
            subset=False
            rest_at=set()
            for t in atom_lst:
                if set(t)!=t1:
                    rest_at.update(t)
            if t1<rest_at:
                subset=True
            if not subset:
                atom_lst2.append(list(t1))
            else:
                atom_lst.remove(atom_lst3[m])
        return atom_lst2
    def get_ring_bond(self,mol):
        bis = mol.GetSubstructMatches(Chem.MolFromSmarts('[!R][R]'))
        bs = [mol.GetBondBetweenAtoms(x,y).GetIdx() for x,y in bis]
        return bs
    def check_bond(self,mol):
        ring_bonds=self.get_ring_bond(mol)
        if len(ring_bonds)>0:
            nm2 = Chem.FragmentOnBonds(mol, ring_bonds , addDummies=False)
            mol_list = Chem.GetMolFrags(nm2, asMols=True)
            at_lst=set()
            for m in mol_list:
                at_lst.update([Chem.MolToSmarts(m)])
            return at_lst
        else:
            return None
    def check(self,query,alllst):
        subset=False
        for m in range(len(alllst)):
            t1=set(alllst[m])
            if t1>=query:
                subset=True
                break
        return subset
    fg=['[N;D2]-[C;D3](=O)-[C;D1;H3]','C(=O)[O;D1]','C(=O)[O;D2]-[C;D1;H3]','C(=O)-[C;D1]','C(=O)-[N;D1]',
            'C(=O)-[C;D1;H3]','[N;D2]=[C;D2]=[O;D1]','[N;D2]=[C;D2]=[S;D1]','[N;D3](=[O;D1])[O;D1]','[N;R0]=[O;D1]',
            '*=[N;R0]-[O;D1]','*=[N;R0]-[C;D1;H3]','[N;R0]=[C;D1;H2]','[N;D2]=[N;D2]-[C;D1;H3]','[N;D2]=[N;D1]',
            '[N;D2]#[N;D1]','[C;D2]#[N;D1]','[S;D4](=[O;D1])(=[O;D1])-[N;D1]','[N;D2]-[S;D4](=[O;D1])(=[O;D1])-[C;D1;H3]',
            '[S;D4](=O)(=O)-[O;D1]','[S;D4](=O)(=O)-[O;D2]-[C;D1;H3]','[S;D4](=O)(=O)-[C;D1;H3]','[S;D4](=O)(=O)-[Cl]',
            '[S;D3](=O)-[C;D1]','[S;D2]-[C;D1;H3]','[S;D1]','*=[S;D1]','[#9,#17,#35,#53]','[C;D4]([C;D1])([C;D1])-[C;D1]',
            '[C;D4](F)(F)F ','[C;D2]#[C;D1;H]','[C;D3]1-[C;D2]-[C;D2]1 ','[O;D2]-[C;D2]-[C;D1;H3]','[O;D2]-[C;D1;H3]',
            '[O;D1]','*=[O;D1]','[N;D1]','*=[N;D1]','*#[N;D1]']
    fg_mols=[Chem.MolFromSmarts(x) for x in fg]
    def get_fg(self,mol,fg_mols=fg_mols):
        atom_lst=[]
        for fgm in fg_mols:
            if len(mol.GetSubstructMatches(fgm)) > 0:
                atom_lst.extend(mol.GetSubstructMatches(fgm))
        atom_lst2=self.rm_small_frag(atom_lst)
        return atom_lst,atom_lst2
    def get_fragment(self):
        m = Chem.MolFromSmiles(self.smiles)
        atom_lst,atom_lst2=self.get_fg(m)
        fcatom=[]
        for n in atom_lst2:
            fcatom.extend(n)
        fcatom=set(fcatom)
        cut_bond=[]
        for bond in m.GetBonds():
            a1=bond.GetBeginAtomIdx()
            a2=bond.GetEndAtomIdx()
            t=set([a1,a2])&fcatom
            if len(t)==1:
                cut_bond.append(bond.GetIdx())
            elif len(t)==2:
                if not self.check(set([a1,a2]),atom_lst2):
                    cut_bond.append(bond.GetIdx())
            else:
                continue
        cut_bond.extend(self.get_ring_bond(m))
        cut_bond=list(set(cut_bond))
        if len(cut_bond) > 0:
            try:
                mol = Chem.MolFromSmiles(self.smiles)
                nm = Chem.FragmentOnBonds(mol, cut_bond , addDummies=False)
                mol_list = Chem.GetMolFrags(nm, asMols=True)
            except:
                mol = Chem.MolFromSmiles(self.smiles,sanitize=False)
                nm = Chem.FragmentOnBonds(mol, cut_bond , addDummies=False)
                mol_list = Chem.GetMolFrags(nm, asMols=True)
        else:
            mol_list=[Chem.MolFromSmiles(self.smiles)]
        reg1 = re.compile('\[\d+\*\]{1}')
        allfrags=set()
        bondatoms=set()
        b= Chem.MolFromSmarts('[R][R]')
        for bs in m.GetSubstructMatches(b):
            bondatoms.update(bs)
        fc_bat=fcatom&bondatoms
        for ba in fc_bat:
            bondatoms.remove(ba)
        for nn in mol_list:
            t=BRICS.BRICSDecompose(nn)
            for f in t:
                fsm=re.sub(reg1,'',f).replace('()','')
                fm=Chem.MolFromSmiles(fsm)
                if fm is None:
                    continue
                c_bond=self.check_bond(fm)
                if c_bond:
                    allfrags.update(c_bond)
                else:
                    fsm=Chem.MolToSmarts(fm)
                    allfrags.update([fsm])
        all_lst=[]
        frag_smi={'O=[N+][O-]':'O=NO','[#6]1:[#7]:[#6]:[#6]:[#7H]:1':'[#6]1:[#7]:[#6]:[#6]:[#7]:1'}
        for frag in allfrags:
            if frag in frag_smi.keys():
                frag=frag_smi[frag]
            patt = Chem.MolFromSmarts(frag)
            all_lst.extend(m.GetSubstructMatches(patt))
        all_lst2=[]
        for f in all_lst:
            t=set(f)&fcatom
            tb=set(f)&bondatoms
            if len(t)==0 and len(tb)==0 :
                all_lst2.append(f)
            elif set(f)<=bondatoms:
                all_lst2.append(f)
            else:
                if self.check(set(f),atom_lst):
                    all_lst2.append(f)
        all_lst3=self.rm_small_frag(all_lst2)
        all_lst4=self.rm_small_frag2(all_lst3)
        return all_lst4 

