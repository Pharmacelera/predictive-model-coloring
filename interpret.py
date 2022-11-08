#!/usr/bin/env python

import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.Draw import SimilarityMaps
import matplotlib.pyplot as plt
from pathlib import Path
import os
import copy
import joblib
import re
from sklearn.preprocessing import MinMaxScaler
import ast
import matplotlib.ticker as mticker
import matplotlib.colors as colors
from matplotlib import cm
import math
from matplotlib.colors import LinearSegmentedColormap
from rdkit import RDLogger   
RDLogger.DisableLog('rdApp.*')
import warnings
from fragment import Get_Fragment_lst

#
# Copyright (c) 2021 Pharmacelera S.L.
# All rights reserved.
#
# Description: Script for interpretation by atom/fragmnet coloring
#
# Usage: Define the molecule featurized method and model prediction method and run this script
#



class Interpret_image():    
    """Script for interpretation by atom/fragmnet coloring.

    Parameters
    ----------
    predict: a defined model prediction method.
    featurize_mol: a defined molecule featurized method for molecule containing no dummy atoms.
    featurize_mol_dummy: a defined molecule featurized method for molecule containing dummy atoms. Default is None.
    scaler : a defined scaler method for descriptors. Default is None
    **kwargs
        additional and unused keyword arguments
    """

    def __init__(self,predict,featurize_mol,featurize_mol_dummy=None,scaler=None,**kwargs,):
        self.featurize_mol = featurize_mol
        if featurize_mol_dummy is None:
            self.featurize_mol_dummy = featurize_mol
        else:
            self.featurize_mol_dummy = featurize_mol_dummy
        self.predict = predict
        self.scaler = scaler
        
    def keep_bonds(self,smi2):
        """set bond types"""
        b1 = []
        mol2 = Chem.MolFromSmiles(smi2)
        try:
            mol1 = Chem.MolFromSmiles(smi2.replace('*','C'))
            for b in mol1.GetBonds():
                b1.append(b.GetBondType())
        except:
            mol1= Chem.MolFromSmiles(smi2.replace('*','c'))
            for b in mol1.GetBonds():
                b1.append(b.GetBondType())
        for b in mol2.GetBonds():
            tmp = b.GetBondType()
            if tmp in [Chem.rdchem.BondType.ZERO,Chem.rdchem.BondType.UNSPECIFIED]:
                continue
            elif tmp == b1[b.GetIdx()]:
                continue
            else:
                b.SetBondType(b1[b.GetIdx()])

        for bond in mol2.GetBonds():
            if (bond.GetBeginAtom().GetAtomicNum()==0) and (bond.GetEndAtom().GetAtomicNum()==0 ):
                bond.SetBondType(Chem.rdchem.BondType.ZERO)
            elif (bond.GetBeginAtom().GetAtomicNum()==0) or (bond.GetEndAtom().GetAtomicNum()==0 ):
                bond.SetBondType(Chem.rdchem.BondType.SINGLE)           	
        mol2.UpdatePropertyCache(strict=False)
        #mol2=Chem.MolFromSmiles(Chem.MolToSmiles(mol2))
        return mol2

    def setdummybond(self,mol,lst):
        """set dummy atoms"""
        for bond in mol.GetBonds():
            if (bond.GetBeginAtomIdx() in lst) or (bond.GetEndAtomIdx() in lst):
                bond.SetBondType(Chem.rdchem.BondType.SINGLE)
        for idx in lst:
            mol.GetAtomWithIdx(idx).SetAtomicNum(0)
        smile=Chem.MolToSmiles(mol)
        mol=self.keep_bonds(smile)
        return mol

    def get_contribution(self,mol,frag_lst):
        """compute the atoms or fragments contribution"""
        X0=self.featurize_mol(mol)
        if self.scaler is not None:
                X0=self.scaler(X0)
        ypred=self.predict(X0)[0]
        weightdf = []
        for lst in frag_lst:
            mol2=copy.deepcopy(mol)
            tmpmol=self.setdummybond(mol2,lst)
            try:
                x=self.featurize_mol_dummy(tmpmol)
            except:
                x=self.featurize_mol_dummy(Chem.MolFromSmiles(Chem.MolToSmiles(tmpmol)))
            if self.scaler is not None:
                x=self.scaler(x)
            pred_tmp=self.predict(x)[0]
            delta=round(ypred - pred_tmp,2)
            weightdf.append(delta)
        weighlst=dict()
        for index,i in enumerate(frag_lst):
            try:
                i = ast.literal_eval(i)
            except:
                if type(i) != list:
                    i=[i]
            for n in i:
                if n not in weighlst.keys():
                    weighlst[int(n)]=weightdf[index]
        for n in range(mol.GetNumAtoms()):
            if n not in weighlst.keys():
                weighlst[int(n)]=0
        weighlst=dict(sorted(weighlst.items()))
        weightdf=np.array(list(weighlst.values())).reshape(-1,1)
        if weightdf.min() >= 0:
            min_max_scaler = MinMaxScaler((0,1))
        elif  weightdf.max() <= 0:
            min_max_scaler = MinMaxScaler((-1,0))
        else:
            min_max_scaler =MinMaxScaler((-1,1))
        weightdf_norm=min_max_scaler.fit_transform(weightdf).reshape(1,-1)[0]
        return weightdf,weightdf_norm,ypred

    def get_image(self,smiles,label,im_path='.',level='frag',frag_lst=None,colorMap=None,scale=-1,sigma=None,**kwargs,):
        """produce interpretation image
        Parameters
        ----------
        smiles : a single SMILES for interpretation.
        label : label of the molecule.
        im_path : path to save images. Default is current path.
        level : level of interpretation. atom: Atom level. frag: fragment level defeind by Get_Fragment_lst method. self-defined: defeind list by users. Default is 'frag'.
        frag_lst : need to provide a list for interpretation when in 'self-defined' level. e.g. [[1],[2],[3,4,5],[6,7]]. Default is None.
        colorMap : users could set a colormap of matplotlib for coloring atoms. Default is None which use 'PiYG'.
        featurize_mol_dummy: a defined molecule featurized method for molecule containing dummy atoms. Default is None.
        scaler : scaler weights defined by RDKIT. Default is -1.
        sigma : sigma value in Draw.calcAtomGaussians defined by RDKIT. Default is None.
        **kwargs
        additional and unused keyword arguments
        
        Return
        ----------
        One original image and one normalized image for interpretation. 
        """
        if level != 'self-defined':
            frag_lst=Get_Fragment_lst(smiles,level).mol_list
        mol=Chem.MolFromSmiles(smiles)
        weightdf,weightdf_norm,ypred=self.get_contribution(mol,frag_lst)
        print('Predicted value of %s : %.2f '%(label,ypred))
        fig0=SimilarityMaps.GetSimilarityMapFromWeights(mol,weightdf)
        ypred=str(round(ypred,2)).replace('.','_')
        Path(im_path).mkdir(parents=True,exist_ok=True)
        if os.path.exists(im_path+'/'+label+'_'+str(ypred)+'_origin.png'):
            os.remove(im_path+'/'+label+'_'+str(ypred)+'_origin.png')
        fig0.savefig(im_path+'/'+label+'_'+str(ypred)+'_origin.png', bbox_inches='tight')
        plt.close(fig0)
        fig=Draw.MolToMPL(mol)
        cmap = cm.seismic
        if sigma is None:
            if mol.GetNumBonds() > 0:
                bond = mol.GetBondWithIdx(0)
                idx1 = bond.GetBeginAtomIdx()
                idx2 = bond.GetEndAtomIdx()
                sigma = 0.3 * math.sqrt(
                sum([(mol._atomPs[idx1][i] - mol._atomPs[idx2][i])**2 for i in range(2)]))
            else:
                sigma = 0.3 * math.sqrt(sum([(mol._atomPs[0][i] - mol._atomPs[1][i])**2 for i in range(2)]))
            sigma = round(sigma, 2)
        x,y,z=Draw.calcAtomGaussians(mol,sigma,step=0.01,weights=weightdf_norm)
        if scale <= 0.0:
            maxScale = max(math.fabs(np.min(z)), math.fabs(np.max(z)))
        else:
            maxScale = scale
        if colorMap is None:
            if cm is None:
                raise RuntimeError("matplotlib failed to import")
            PiYG_cmap = cm.get_cmap('PiYG', 2)
            cmap = LinearSegmentedColormap.from_list(
                'PiWG', [PiYG_cmap(0), (1.0, 1.0, 1.0), PiYG_cmap(1)], N=255)
        im=fig.axes[0].imshow(z, interpolation='nearest', extent=(0, 1, 0, 1),
                    cmap=cmap, origin='lower',alpha=0.5,vmin=-maxScale, vmax=maxScale)
        label_format = '{:,.2f}'
        cbar=plt.colorbar(im,ax=fig.axes[0])
        ticks_loc = cbar.ax.get_yticks().tolist()
        cbar.ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        tt=cbar.ax.set_yticklabels([label_format.format(x) for x in np.linspace(-1,1,len(ticks_loc))],fontsize=14)
        if os.path.exists(im_path+'/'+label+'_'+str(ypred)+'_normalized.png'):
            os.remove(im_path+'/'+label+'_'+str(ypred)+'_normalized.png')
        fig.savefig(im_path+'/'+label+'_'+str(ypred)+'_normalized.png', bbox_inches='tight')
        plt.close(fig)

