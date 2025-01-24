import xlsxwriter
import cairosvg
import pandas as pd
import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ColorConverter
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import MolToImage, DrawMorganBit

def AddHeadFormat(workbook):
    """
    Add a style to the workbook that is used in the header of the table
    """
    format = workbook.add_format(dict(bold=True, align='center', valign='vcenter', size=12))
    format.set_bg_color('white')
    format.set_border_color('black')
    format.set_border()
    return format

def AddDataFormat(workbook):
    """
    Add a style to the workbook data place 
    """
    format = workbook.add_format(dict(bold=False, align='center', valign='vcenter', size=15))
    format.set_text_wrap()
    format.set_bg_color('white')
    format.set_border_color('black')
    format.set_border()
    
    return format

def writeImageToFile(mol, ofile, width=300, height=300):
    """
    write molecule image to a file with formatting options
    """       
    img = MolToImage(mol, (width, height))
    img.save(ofile, bitmap_format='png')


def WriteDataFrameSmilesToXls(pd_table, smiles_colnames, out_filename,  smirks_colnames=None, max_mols=10000, 
                            retain_smiles_col=False, use_romol=False):
    """
    Write panads DataFrame containing smiles as molcular image
	rdkit version...

    input:
    ------
    pd_table:
    smiles_colnames: must set the smiles column names where smiles are converted to images
    max_mol: For avoid generating too much data 
    out_filename: output file name 
    smirks_colname: reaction smiles (smirks) column name which is decomposed to left and right parts to visualization
    retain_smiles_col: (retaining SMIELS columns or mot )
    output:
    ------
    None: 

    """
    if isinstance(smiles_colnames, str):
        smiles_colnames = [smiles_colnames]
    
    if smiles_colnames is None:
        smiles_colnames = ['']

    if use_romol:
        pd_table[smiles_colnames] = pd_table[smiles_colnames].map(lambda x: Chem.MolToSmiles(x) if x is not None else '') 
    
    if retain_smiles_col:
        pd_smiles = pd_table[smiles_colnames].copy()
        pd_smiles.columns = ['{}_SMI'.format(s) for s in smiles_colnames]
        pd_table = pd.concat([pd_table, pd_smiles], axis=1)

    if smirks_colnames is not None:
        if isinstance(smirks_colnames, str):
            smirks_colnames = [smirks_colnames]
        for smirks_col in smirks_colnames:
            lname, midname, rname   = f'left_{smirks_col}', f'middle_{smirks_col}', f'right_{smirks_col}'
            pd_table[lname]         = pd_table[smirks_col].str.split('>').str[0]
            pd_table[midname]       = pd_table[smirks_col].str.split('>').str[1]
            pd_table[rname]         = pd_table[smirks_col].str.split('>').str[2]
            
            # check the middle part (if no condition for all the smirks, remove it)
            if (pd_table[midname]=='').all():
                del pd_table[midname]
                smirks_names = [lname, rname]
            else:
                smirks_names = [lname, midname, rname]
            smiles_colnames.extend(smirks_names)

    # if the column contain objects then it convers to string
    array_columns = [col for col in pd_table.columns if isinstance(pd_table[col].iloc[0], np.ndarray)]
    pd_table[array_columns] = pd_table[array_columns].map(lambda x: str(x))

    # set up depiction option
    width, height = 250, 250
   
    if not isinstance(pd_table, pd.DataFrame):
        raise ValueError("pd_table must be pandas DataFrame")
    
    if len(pd_table) > max_mols:
        raise ValueError('maximum number of rows is set to %d but input %d' %(max_mols, len(pd_table)))

    workbook = xlsxwriter.Workbook(out_filename)
    worksheet = workbook.add_worksheet()

    # Set header to a workbook
    headformat = AddHeadFormat(workbook)
    dataformat = AddDataFormat(workbook)

    # Estimate the width of columns
    maxwidths = dict()
    
    if not pd_table.index.name:
        pd_table.index.name = 'index'
    
    for column in pd_table:
        if column in smiles_colnames: # for structure column
            maxwidths[column] = width *0.15 # I do not know why this works
        else:
            if pd_table[column].dtype == list: # list to str
                pd_table[column] = pd_table[column].apply(str)
            l_txt = pd_table[column].apply(lambda x: len(str(x)))
            
            l_len = np.max(l_txt)
            l_len = max(l_len, len(str(column)))
            maxwidths[column] = l_len * 1.2  # misterious scaling
    
    # Generate header (including index part)
    row, col = 0, 0
    worksheet.set_row(row, None, headformat)
    worksheet.set_column(col, col, len(str(pd_table.index.name)))
    worksheet.write(row, col, pd_table.index.name)
    
    for colname in pd_table:
        col +=1
        worksheet.set_column(col, col, maxwidths[colname])
        worksheet.write(row, col, colname)

    # temporary folder for storing figs
    with tempfile.TemporaryDirectory() as tmp_dir:

        # Generate the data
        for idx, val in pd_table.iterrows():
            row += 1
            worksheet.set_row(row, height * 0.75, dataformat)

            col = 0
            worksheet.write(row, col, idx)
            
            # contents 
            for cname, item in val.items():
                col += 1
                if cname in smiles_colnames:
                    fname = os.path.join(tmp_dir, '%d_%d.png' %(row, col)) 
                    if isinstance(item, str):
                        mol = Chem.MolFromSmiles(item)
                    else:
                        mol = item
                    if mol is not None:
                        writeImageToFile(mol, fname, int(width*0.9), int(height*0.9))
                        worksheet.insert_image(row, col, fname, dict(object_position=1, x_offset=1, y_offset=1))
                else:
                    try:
                        worksheet.write(row, col, item)
                    except: 
                        continue
        workbook.close()

def WriteDataFrameSVGToXls(pd_table, SVG_colnames, out_filename,
                            retain_SVG_col=False):
    """
    Author : Yuto IWASAKI ; Create : 2023-11-06 ; Last-update : 2023-11-06

    Write panads DataFrame containing SVG strings
    input:
    ------
    pd_table:
    SVG_colnames: must set the SVG strings column names where strings are converted to images
    out_filename: output file name
    retain_SVG_col: (retaining SVG string columns or not)
    output:
    ------
    None: 

    """
    if isinstance(SVG_colnames, str):
        SVG_colnames = [SVG_colnames]
    
    if SVG_colnames is None:
        SVG_colnames = ['']

    # pd_table[SVG_colnames] = pd_table[SVG_colnames].map(
    #     lambda x: cairosvg.svg2png(x) if x is not None else '') 
    
    if retain_SVG_col:
        pd_smiles = pd_table[SVG_colnames].copy()
        pd_smiles.columns = ['{}_string'.format(s) for s in SVG_colnames]
        pd_table = pd.concat([pd_table, pd_smiles], axis=1)


    # if the column contain objects then it convers to string
    array_columns = [col for col in pd_table.columns if isinstance(pd_table[col].iloc[0], np.ndarray)]
    pd_table[array_columns] = pd_table[array_columns].map(lambda x: str(x))

    # set up depiction option
    width, height = 250, 250
   
    if not isinstance(pd_table, pd.DataFrame):
        raise ValueError("pd_table must be pandas DataFrame")

    workbook = xlsxwriter.Workbook(out_filename)
    worksheet = workbook.add_worksheet()

    # Set header to a workbook
    headformat = AddHeadFormat(workbook)
    dataformat = AddDataFormat(workbook)

    # Estimate the width of columns
    maxwidths = dict()
    
    if not pd_table.index.name:
        pd_table.index.name = 'index'
    
    for column in pd_table:
        if column in SVG_colnames: # for structure column
            maxwidths[column] = width *0.15 # I do not know why this works
        else:
            if pd_table[column].dtype == list: # list to str
                pd_table[column] = pd_table[column].apply(str)
            l_txt = pd_table[column].apply(lambda x: len(str(x)))
            
            l_len = np.max(l_txt)
            l_len = max(l_len, len(str(column)))
            maxwidths[column] = l_len * 1.2  # misterious scaling
    
    # Generate header (including index part)
    row, col = 0, 0
    worksheet.set_row(row, None, headformat)
    worksheet.set_column(col, col, len(str(pd_table.index.name)))
    worksheet.write(row, col, pd_table.index.name)
    
    for colname in pd_table:
        col +=1
        worksheet.set_column(col, col, maxwidths[colname])
        worksheet.write(row, col, colname)

    # temporary folder for storing figs
    with tempfile.TemporaryDirectory() as tmp_dir:

        # Generate the data
        for idx, val in pd_table.iterrows():
            row += 1
            worksheet.set_row(row, height * 0.75, dataformat)

            col = 0
            worksheet.write(row, col, idx)
            
            # contents 
            for cname, item in val.items():
                if item==None:
                    continue
                col += 1
                if cname in SVG_colnames:
                    fname = os.path.join(tmp_dir, '%d_%d.png' %(row, col))
                    if isinstance(item,str):
                        cairosvg.svg2png(bytestring=item,write_to=fname)
                    else:
                        item.save(fname)
                    worksheet.insert_image(row, col, fname, dict(object_position=1, x_offset=1, y_offset=1))
                else:
                    try:
                        worksheet.write(row, col, item)
                    except: 
                        continue
        workbook.close()

def bitvisualSVG(smi_and_bit):
    smi_dict = {}
    svg_dict = {}
    for key_smi, bit_dict in smi_and_bit:
        mol = Chem.MolFromSmiles(key_smi)
        for key_bit,bit in bit_dict.items():
            if key_bit not in smi_dict:
                smi_dict[key_bit] = list()
                svg_dict[key_bit] = list()
            for b in bit:
                if b[1]!=0:
                    env = Chem.FindAtomEnvironmentOfRadiusN(mol, b[1], b[0])
                    amap = {}
                    submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                    smi = Chem.MolToSmiles(submol)
                else:
                    env = Chem.FindAtomEnvironmentOfRadiusN(mol, 1, b[0])
                    amap = {}
                    submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                    smi = f'{Chem.MolToSmiles(submol)} <radius=0>'
                if smi not in smi_dict[key_bit]:
                    smi_dict[key_bit].extend([smi])
                    svg_dict[key_bit].extend([DrawMorganBit(mol,key_bit,bit_dict,
                                                            )])
    return svg_dict

def BitExplanation(smi_and_bit,savename='bitexp.xlsx'):
    svg_dict = bitvisualSVG(smi_and_bit)
    df_bit = pd.DataFrame(svg_dict.values(), index=svg_dict.keys()).sort_index()
    WriteDataFrameSVGToXls(df_bit,SVG_colnames=df_bit.columns,out_filename=savename)

def ConvertSARMatrix2Excel(pd_df, ofname='test.xlsx'):
    """
    (index: core_smiles, columns: substructure) -> index and columns are converted to excel format 
    rdkit ver...

    """
    # Set depict options 
    org_width, org_height = 250, 250
    workbook    = xlsxwriter.Workbook(ofname)
    worksheet   = workbook.add_worksheet()

    #headfont, headformat = AddHeadFormat(workbook)
    dataformat = AddDataFormat(workbook)

    # index is the smiles
    np_val = pd_df.values
    n, d   = np_val.shape
    
    row, col = 0, 0
    width = int(0.15*org_width) # scaling
    height = int(0.75*org_height)

    worksheet.set_row(0, height, dataformat)
    worksheet.set_column(0, d, width)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Row 
        for i, idx in enumerate(pd_df.index):
            mol = Chem.MolFromSmiles(idx)
            fname = os.path.join(tmp_dir, "index_%d.png"%(i))
            if mol is not None:
                writeImageToFile(mol, fname, int(org_width*0.9), int(org_height*0.9))
                worksheet.set_row(i+1, height, dataformat)
                worksheet.insert_image(i+1, 0, fname, dict(object_position=1, x_offset=1, y_offset=1))
              
        # Column
        for i, idx in enumerate(pd_df.columns):
            mol = Chem.MolFromSmiles(idx)
            fname = os.path.join(tmp_dir, 'column_%d.png'%i)
            if mol is not None:
                writeImageToFile(mol, fname, int(org_width*0.9), int(org_height*0.9))
                worksheet.insert_image(0, i+1, fname, dict(object_position=1, x_offset=1, y_offset=1))
              
        # Data (compound Index (?))
        na_table = pd_df.isna()
        for i in range(n):
            for j in range(d):
                if not na_table.iloc[i,j]:
                    try:
                        worksheet.write(i+1, j+1, pd_df.iloc[i,j], dataformat)
                    except:
                        continue
        workbook.close()

def boxplotAsFixedColor(values,labels:list=[None],colors:list=[None],alphas:list=[None],figsize:tuple=(12,10),ax=[None]):
    """
    Author : Yuto IWASAKI ; Create : 2023-11-06 ; Last-update : 2023-11-06

    Boxplot with fixed color(s)
    input:
    ------
    values: Allay or list. Must be 1- ~ 3-dimension.
    labels: Allay or list. Must be (dim(values) - 1)-dimensional
    colors: Allay or list. Must be 1-dimensional
    alphas: Allay or list. Must be 1-dimensional
    figsize: tuple with (horizon, vertical). If 'ax' is given, this option will not work.
    ax: matplotlib.pyplot.axes object. If this is given, return will be plotted ax.
    output:
    ------
    fig (unless 'ax' is given) or ax ('ax' is given)

    """
    assert(np.ndim(values)<4 and np.ndim(values)>0)
    while np.ndim(values)<3:
        values = [values]
        labels = [labels]
    assert(len(values)==len(labels) and len(labels)==len(colors) and len(colors)==len(alphas))
    ret_ax = True
    if ax==None:
        ret_ax = False
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1)
    for val, lab, col, alp in zip(values, labels, colors, alphas):
        ax.boxplot(val,
                   labels=lab,
                   patch_artist=True,
                   boxprops={'facecolor' : col,
                             'color' : col,
                             'alpha' : alp},
                   medianprops={'color' : col,
                             'alpha' : alp},
                   whiskerprops={'color' : col,
                             'alpha' : alp},
                   capprops={'color' : col,
                             'alpha' : alp},
                   flierprops={'markeredgecolor' : col,
                             'alpha' : alp})
    if ret_ax : return ax
    return fig


def CompoundsVisualizerWithIsotopeReactionCenters(smls):
    # gray = ColorConverter.to_rgb('lightgray')

    mols = [Chem.MolFromSmiles(m) for m in smls]
    # hl_lists = []

    # for mol in mols:
    #     hl_list = []
    #     for atom in mol.GetAtoms():
    #         if int(atom.GetIsotope()) >= 900:
    #             hl_list.append(atom.GetIdx())
    #     hl_lists.append(hl_list)

    options = Draw.MolDrawOptions()
    options.useBWAtomPalette()
    options.maxFontSize = -1
    options.fixedFontSize = 50
    options.fixedBondLength = 65
    # options.baseFontSize = 6
    options.bondLineWidth = 6
    # options.isotopeLabels = False
    # options.highlightRadius = 1
    # options.setHighlightColour(gray)

    img = Draw.MolsToGridImage(mols, molsPerRow=3,
                            subImgSize=(1600,800),
                            # highlightAtomLists=hl_lists,
                            drawOptions=options)
    return img


def CompoundsVisualizerWithHighlightReactionCenters(smls, molsPerRow=3, subImgSize=(1600,800), ofile=None):
    pink    = ColorConverter.to_rgb('pink')
    skyblue = ColorConverter.to_rgb('skyblue')
    
    isotope_colors = {
        1000: pink,  # ReactionCenter
        900: skyblue   # LeavingGroup
    }

    mols = [Chem.MolFromSmiles(m) for m in smls]
    hl_idx_lists = []
    hl_clr_lists = []

    for mol in mols:
        hl_idx = []
        hl_clr = {}
        for atom in mol.GetAtoms():
            isotope = int(atom.GetIsotope())
            if isotope >= 900:
                idx = atom.GetIdx()
                hl_idx.append(idx)
                hl_clr[idx] = isotope_colors[isotope]
        hl_idx_lists.append(hl_idx)
        hl_clr_lists.append(hl_clr.copy())

    options = Draw.MolDrawOptions()
    options.useBWAtomPalette()
    options.maxFontSize = -1
    options.fixedFontSize = 100
    options.fixedBondLength = 65
    # options.baseFontSize = 6
    options.bondLineWidth = 6
    options.isotopeLabels = False
    options.highlightRadius = 1
    # options.setHighlightColour(gray)

    img = Draw.MolsToGridImage(mols, molsPerRow=molsPerRow,
                            subImgSize=subImgSize,
                            highlightAtomLists=hl_idx_lists,
                            highlightAtomColors=hl_clr_lists,
                            drawOptions=options)
    if ofile is not None:
        img.save(ofile)
        del img
        return
    return img

if __name__=='__main__':
    # img = CompoundsVisualizerWithIsotopeReactionCenters(['[1000NH2]CCOc1cccc2[nH]ccc12.O=C(c1ccc(F)c(Cl)c1)N1CCC(F)([1000CH]=[900O])CC1','O=C(c1ccc(F)c(Cl)c1)N1CCC(F)(CNCCOc2cccc3[nH]ccc23)CC1','[1000NH2]CCCOc1cccc2[nH]ccc12.O=C1COc2ccc(OC[1000CH]=[900O])cc2N1','O=C(c1ccc(F)c(Cl)c1)N1CCC(F)(CNCCOc2ccccc2Cl)CC1'])
    # img = CompoundsVisualizerWithIsotopeReactionCenters(['COc1ccc(-c2ccccc2)cc1N1CCN(CCn2cnc3sc4c(c3c2=O)CCN(C)C4)CC1','CN1CCc2c(sc3ncn(C[1000CH2][900Cl])c(=O)c23)C1','COc1ccc(-c2ccccc2)cc1N1CC[1000NH]CC1'])
    # img = CompoundsVisualizerWithHighlightReactionCenters(['c1ccc(NC(=O)CN(C)C)cc1CC1CCN(CSCOc2ccc3OC(=O)C=C(C)(c3c2))CC1','CN1CCc2c(sc3ncn(C[1000CH2][900Cl])c(=O)c23)C1','COc1ccc(-c2ccccc2)cc1N1CC[1000NH]CC1'])
    # img = CompoundsVisualizerWithHighlightReactionCenters(['O=C1c2cccc3cccc(c23)N1CCCCCCN1CCN(c2ccc(Br)cc2)CC1','[900Cl][1000CH2]CCCCCN1CCN(c2ccc(Br)cc2)CC1','O=C1[1000NH]c2cccc3cccc1c23','O=C1c2cccc3cccc(c23)N1CCCCC[1000CH2][900Cl]','Brc1ccc(N2CC[1000NH]CC2)cc1'])
    # img = CompoundsVisualizerWithHighlightReactionCenters(['CC(C)C[C@H]1C(=O)N2CCC[C@H]2[C@]3(N1C(=O)[C@](O3)(C(C)C)NC(=O)[C@H]4CN([C@@H]5CC=6C7=C(C=CC=C7NC6Br)C5=C4)C)O'])
    # img = CompoundsVisualizerWithHighlightReactionCenters(['CCCCCCC(C)(C)c1cc2c([1000c]([1000O]Cc3ccccc3)c1)-c1nn(CC)cc1C(C)(C)O2','CCCCCCC(C)(C)c1cc2c([1000c]([900Cl])c1)-c1nn(CC)cc1C(C)(C)O2.[1000OH]Cc1ccccc1','[900Cl][1000c]1ccc(N2CCN(c3ncnc4c3nc3n4CCCCC3)CC2)cc1.Cc1cc(C(C)([1000OH])C(F)(F)F)ccc1Cl'])
    img = CompoundsVisualizerWithHighlightReactionCenters(['Fc1ccc(S(=O)(N(C2CC[1000N]([1000CH2]SCOC3=CC=C(C(C)=CC(O4)=O)C4=C3)CC2)C)=O)cc1','Fc1ccc(S(=O)(N(C)C2CC[1000NH]CC2)=O)cc1','[900Cl][1000CH2]SCOC3=CC=C4C(C)=CC(OC4=C3)=O'])
    # img = CompoundsVisualizerWithHighlightReactionCenters(['c1ccc(OCCCN2CCN(c3ccccc3)CC2)cc1',
    #                                                        'c1ccc(OCCCCN2CCN(c3ccccc3)CC2)cc1',
    #                                                        'c1ccc(CCN2CCN(c3ccccc3)CC2)cc1',
    #                                                        'O=C(CCN1C(=O)COc2ccccc21)Nc1ccccc1OCC(=O)c1ccc2c(c1)NC(=O)CO2',
    #                                                        'O=C(CCN1CCCc2ccccc21)Nc1ccccc1OCC(=O)c1ccc2c(c1)NC(=O)CO2',
    #                                                        'O=C1COc2ccc(C(=O)COc3cccc(C(=O)NCCCN4C(=O)COc5ccccc54)c3)cc2N1',
    #                                                        'Fc1ccc(S(=O)(N(C2CCN(CSCOC3=CC=C(C(C)=CC(O4)=O)C4=C3)CC2)C)=O)cc1',],subImgSize=(1800,1000))
    # img = CompoundsVisualizerWithHighlightReactionCenters(['O=C(c1ccc(F)c(Cl)c1)N1CCC(F)(CNCCOc2cccc3[nH]ccc23)CC1','O=C(c1ccc(F)c(Cl)c1)N1CCC(F)(CNCCOc2ccccc2Cl)CC1','[1000NH2]CCOc1cccc2[nH]ccc12.O=C(c1ccc(F)c(Cl)c1)N1CCC(F)([1000CH]=[900O])CC1','[1000NH2]CCOc1ccccc1Cl.O=C(c1ccc(F)c(Cl)c1)N1CCC(F)([1000CH]=[900O])CC1','[1000NH2]CCOc1ccccc1Cl.O=C(c1ccc(F)c(Cl)c1)N1CCC(F)([1000CH]=[900O])CC1'],molsPerRow=2,subImgSize=(1800,1000))
    img.save('fig.png')