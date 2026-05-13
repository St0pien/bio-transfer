DOWNSTREAM_TARGETS = {
    "BACE1": "CHEMBL4822",
    "TYK2": "CHEMBL3553",
    "A2a": "CHEMBL251",
}

UPSTREAM_TARGETS = {
    "JAK1": "CHEMBL2835",
    "JAK2": "CHEMBL2971",
    "JAK3": "CHEMBL2148",
    "EGFR": "CHEMBL203",
    "CDK2": "CHEMBL301",
    "p38alpha": "CHEMBL260",
    "BACE2": "CHEMBL2525",
    "HGFR": "CHEMBL3717",
    "HDAC6": "CHEMBL1865",
    "Cathepsin_D": "CHEMBL2581",
    "Renin": "CHEMBL286",
    "HSP90": "CHEMBL3880",
    "Pepsin": "CHEMBL3295",
    "A1": "CHEMBL226",
    "PDE46": "CHEMBL254",
    "A2b": "CHEMBL255",
    "EDNRA": "CHEMBL252",
    "A3": "CHEMBL256",
    "DRD2": "CHEMBL217",
    "5HT2A": "CHEMBL224",
    "HDAC1": "CHEMBL325",
    "PARP1": "CHEMBL3105",
    "HSP90": "CHEMBL4303",
    "CA2": "CHEMBL205",
    "Prothrombin": "CHEMBL204",
    "AChE": "CHEMBL220",
    "DPP4": "CHEMBL284",
    "COX-1": "CHEMBL221",
    "KCNH2": "CHEMBL240",
    "TRPV1": "CHEMBL4794",
    "ER-alpha": "CHEMBL206",
    "DAT": "CHEMBL238",
    "PTP-1B": "CHEMBL335",
    "H1": "CHEMBL231",
    "H2": "CHEMBL1941",
    "H4": "CHEMBL3759",
}

BACE_1_SIMILARS = ["BACE2", "Pepsin", "Renin", "Cathepsin_D"]
TYK_2_SIMILARS = ["JAK1", "JAK2", "JAK3", "EGFR", "CDK2", "p38alpha"]
A2A_SIMILARS = ["A1", "A2b", "A3", "DRD2", "5HT2A"]

SIMILARS = BACE_1_SIMILARS + TYK_2_SIMILARS + A2A_SIMILARS

ATOM_TYPES = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "P",
    "Cl",
    "Br",
    "I",
]
