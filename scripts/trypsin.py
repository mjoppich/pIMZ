from rpg import enzyme, rule

def get_trypsin():
    # Trypsin
    # https://web.expasy.org/peptide_cutter/peptidecutter_enzymes.html#Tryps
    # RULES: after K except if next aa is P. This rule doesn't apply if W is before K
    # RULES: after R except if next aa is P. This rule doesn't apply if M is before R
    # RULES: don't cleaves CKD, DKD, CKH, CKY, CRK, RRH nor RRR
    # Other way to see it: cleaves after K|R except if P after, but cleaves WKP and MRP. Don't cleaves CKD, DKD, CKH, CKY, CRK, RRH nor RRR
    ENZ = []

    # Cutting rules
    AFTER_K = rule.Rule(0, "K", True, 1) # Always cleaves after K, except...
    AFTER_R = rule.Rule(0, "R", True, 1) # Always cleaves after R, except...

    # Exceptions
    EXCEPT_KP = rule.Rule(1, "P", False, -1) # Never cleaves after K followed by P, except...
    EXCEPT_KD = rule.Rule(1, "D", True, -1) # Always cleaves after K followed by D, except...
    EXCEPT_KH = rule.Rule(1, "H", True, -1) # Always cleaves after K followed by H, except...
    EXCEPT_KY = rule.Rule(1, "Y", True, -1) # Always cleaves after K followed by Y, except...

    EXCEPT_RP = rule.Rule(1, "P", False, -1) # Never cleaves after R followed by P, except...
    EXCEPT_RK = rule.Rule(1, "K", True, -1) # Always cleaves after R followed by K, except...
    EXCEPT_RH = rule.Rule(1, "H", True, -1) # Always cleaves after R followed by H, except...
    EXCEPT_RR = rule.Rule(1, "R", True, -1) # Always cleaves after R followed by R, except...

    # Counter-exceptions
    UNEXCEPT_WKP = rule.Rule(-1, "W", True, -1) # Always cleaves after K followed by P and preceded by W
    UNEXCEPT_CKD = rule.Rule(-1, "C", False, -1) # Never cleaves after K followed by D and preceded by C
    UNEXCEPT_DKD = rule.Rule(-1, "D", False, -1) # Never cleaves after K followed by D and preceded by D
    UNEXCEPT_CKH = rule.Rule(-1, "C", False, -1) # Never cleaves after K followed by H and preceded by C
    UNEXCEPT_CKY = rule.Rule(-1, "C", False, -1) # Never cleaves after K followed by Y and preceded by C

    UNEXCEPT_MRP = rule.Rule(-1, "M", True, -1) # Always cleaves after R followed by P and preceded by M
    UNEXCEPT_CRK = rule.Rule(-1, "C", False, -1) # Never cleaves after R followed by K and preceded by C
    UNEXCEPT_RRH = rule.Rule(-1, "R", False, -1) # Never cleaves after R followed by H and preceded by R
    UNEXCEPT_RRR = rule.Rule(-1, "R", False, -1) # Never cleaves after R followed by R and preceded by R

    # Add counter-exceptions to exceptions
    EXCEPT_KP.rules.append(UNEXCEPT_WKP)
    EXCEPT_KD.rules.append(UNEXCEPT_CKD)
    EXCEPT_KD.rules.append(UNEXCEPT_DKD)
    EXCEPT_KH.rules.append(UNEXCEPT_CKH)
    EXCEPT_KY.rules.append(UNEXCEPT_CKY)

    EXCEPT_RP.rules.append(UNEXCEPT_MRP)
    EXCEPT_RK.rules.append(UNEXCEPT_CRK)
    EXCEPT_RH.rules.append(UNEXCEPT_RRH)
    EXCEPT_RR.rules.append(UNEXCEPT_RRR)

    # Add exception to cutting rules
    AFTER_K.rules.append(EXCEPT_KP)
    AFTER_K.rules.append(EXCEPT_KD)
    AFTER_K.rules.append(EXCEPT_KH)
    AFTER_K.rules.append(EXCEPT_KY)

    AFTER_R.rules.append(EXCEPT_RP)
    AFTER_R.rules.append(EXCEPT_RK)
    AFTER_R.rules.append(EXCEPT_RH)
    AFTER_R.rules.append(EXCEPT_RR)

    # Add rules to enzyme
    ENZ.append(AFTER_K)
    ENZ.append(AFTER_R)

    return enzyme.Enzyme(1, "Trypsin", ENZ, 0)
