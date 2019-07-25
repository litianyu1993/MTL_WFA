# -*- coding: utf-8 -*-
# ######### COPYRIGHT #########
#
# Copyright(c) 2016
# -----------------
#
# * LabEx Archimède: http://labex-archimede.univ-amu.fr/
# * Laboratoire d'Informatique Fondamentale : http://www.lif.univ-mrs.fr/
#
# Contributors:
# ------------
#
# * François Denis <francois.denis_AT_lif.univ-mrs.fr>
# * Rémy Eyraud <remy.eyraud_AT_lif.univ-mrs.fr>
# * Denis Arrivault <contact.dev_AT_lif.univ-mrs.fr>
# * Dominique Benielli <dominique.benielli_AT_univ-amu.fr>
#
# Description:
# -----------
#
# SP2Learning is a toolbox in
# python for spectral learning algorithms developped in the context of
# the Sequence PredIction Challenge (SPiCe).
#
# Version:
# -------
#
# * sp2learn version = 1.1.2
#
# Licence:
# -------
#
# License: 3-clause BSD
#
#
# ######### COPYRIGHT #########
"""This module contains the Load class
   The load class load objects from file
"""

import pickle

class Load(object):
    """ A load instance

    :Example:

    >>> from sp2learn import Load
    >>> l = Load('adr').load_Spice_Sample()

    - Input:

    :param string adr: address and name of the loaden file
    :param string type: (default value = 'SPiCe') indicate
           the structure of the file
    """

    def __init__(self, adr, type='SPiCe'):
        self._type = type
        self._adr = adr

    def load_Spice_Sample(self, lrows=[], lcolumns=[],
                             version="classic", partial=False):
        """
        Load a sample from a Spice file and returns a dictionary
            (word,count)

        - Input:

        :param lrows: number or list of rows,
               a list of strings if partial=True;
               otherwise, based on pref if version="classic" or
               "prefix", fact otherwise
        :type lrows: int or list of int
        :param lcolumns: number or list of columns
               a list of strings if partial=True ;
               otherwise, based on suff if version="classic" or "suffix",
               fact otherwise
        :type lcolumns: int or list of int
        :param string version: (default = "classic") version name
        :param boolean partial: (default value = False) build of partial
               if True partial dictionaries are loaded based
               on nrows and lcolumns

        - Output:

        :returns:  nbL , nbEx , dsample , dpref , dsuff  , dfact
        :rtype: int , int , dict , dict , dict  , dict
        """
        adr = self._adr
        type = self._type

        if type == 'SPiCe':

            if partial:
                return self._load_Spice_partial(
                    adr=adr,
                    lrows=lrows, lcolumns=lcolumns,
                    version=version)
            if  not partial:
                return self._load_Spice_Notpartial(adr=adr)

    def _load_Spice_partial(self, adr,
                               lrows=[], lcolumns=[],
                               version='classic'):
        dsample = {}  # dictionary (word,count)
        dpref = {}
        dsuff = {}
        dfact = {}
        f = open(adr, "r")
        line = f.readline()
        l = line.split()
        nbEx = int(l[0])
        nbL = int(l[1])
        line = f.readline()

        if isinstance(lrows, int):
            lrowsmax = lrows
            version_rows_int = True
        else:
            version_rows_int = False
            lrowsmax = lrows.__len__()
        if isinstance(lcolumns, int):
            lcolumnsmax = lcolumns
            version_columns_int = True
        else:
            lcolumnsmax = lcolumns.__len__()
            version_columns_int = False
        lmax = lrowsmax + lcolumnsmax
        while line:
            l = line.split()
            w = () if int(l[0]) == 0 else tuple([int(x) for x in l[1:]])
            if version == "classic":
                dsample[w] = dsample[w] + 1 if w in dsample else 1
            if version == "prefix" or version == "classic":
                # traitement du mot vide pour les préfixes, suffixes et facteurs
                dpref[()] = dpref[()] + 1 if () in dpref else 1
            if version == "suffix" or version == "classic":
                dsuff[()] = dsuff[()] + 1 if () in dsuff else 1
            if version == "factor" or version == "suffix" \
                    or version == "prefix":
                dfact[()] = dfact[()] + len(w) + 1 if () in dfact else len(w) + 1
            for i in range(len(w)):
                if version == "classic":
                    # dictionaries dpref and dsuff are populated until
                    # respectively lrows and lcolumns
                    if (version_rows_int is True and i + 1 <= lrowsmax) or \
                            (version_rows_int is False and w[:i + 1] in lrows):
                        dpref[w[:i + 1]] = \
                            dpref[w[:i + 1]] + 1 if w[:i + 1] in dpref else 1
                    if (version_columns_int is True and i + 1 <= lcolumnsmax) or\
                        (version_columns_int is False and w[-(i + 1):] in lcolumns):
                        dsuff[w[-(i + 1):]] = dsuff[w[-(i + 1):]] + 1 \
                            if w[-(i + 1):] in dsuff else 1
                if version == "prefix":
                    # dictionaries dpref is populated until
                    # lmax = lrows + lcolumns
                    # dictionaries dfact is populated until lcolumns
                    if ((version_rows_int is True or
                        version_columns_int is True) and
                        i + 1 <= lmax) or\
                        (version_rows_int is False and
                        (w[:i + 1] in lrows)) or\
                        (version_columns_int is False and
                        (w[:i + 1] in lcolumns)):
                        dpref[w[:i + 1]] = dpref[w[:i + 1]] + 1 \
                                if w[:i + 1] in dpref else 1
                    for j in range(i + 1, len(w) + 1):
                        if (version_columns_int is True and (j - i) <= lmax) or \
                            (version_columns_int is False and
                            (w[i:j] in lcolumns )):
                            dfact[w[i:j]] = dfact[w[i:j]] + 1 \
                                 if w[i:j] in dfact else 1
                if version == "suffix":
                    if ((version_rows_int is True or
                        version_columns_int is True) and
                        i  <= lmax) or\
                        (version_rows_int is False and
                        (w[-(i + 1):] in lrows)) or\
                        (version_columns_int is False and
                        (w[-(i + 1):] in lcolumns)):
                        dsuff[w[-(i + 1):]] = dsuff[w[-(i + 1):]] + 1 \
                            if w[-(i + 1):] in dsuff else 1
                    for j in range(i + 1, len(w) + 1):
                        if (version_rows_int is True and (j - i) <= lmax) or \
                            (version_rows_int is False and
                            (w[i:j] in lrows )):
                            dfact[w[i:j]] = dfact[w[i:j]] + 1 \
                                 if w[i:j] in dfact else 1
                if version == "factor":
                    for j in range(i + 1, len(w) + 1):
                        if ((version_rows_int is True or
                            version_columns_int is True) and
                            (j - i) <= lmax) or \
                            (version_rows_int is False and
                            (w[i:j] in lrows)) or \
                            (version_columns_int is False and
                            (w[i:j] in lcolumns)):
                            dfact[w[i:j]] = \
                                dfact[w[i:j]] + 1 if w[i:j] in dfact else 1
            line = f.readline()
        f.close()
        # self._create_pickle_files(adr=adr, dsample=dsample, dpref=dpref,
        #                          dsuff=dsuff, dfact=dfact)
        return nbL, nbEx, dsample, dpref, dsuff, dfact

    def _load_Spice_Notpartial(self, adr):
        dsample = {}  # dictionary (word,count)
        dpref = {}
        dsuff = {}
        dfact = {}
        f = open(adr, "r")
        line = f.readline()
        l = line.split()
        nbEx = int(l[0])
        nbL = int(l[1])
        line = f.readline()
        while line:
            l = line.split()
            w = () if int(l[0]) == 0 else tuple([int(x) for x in l[1:]])
            dsample[w] = dsample[w] + 1 if w in dsample else 1
            # traitement du mot vide pour les préfixes, suffixes et facteurs
            dpref[()] = dpref[()] + 1 if () in dpref else 1
            dsuff[()] = dsuff[()] + 1 if () in dsuff else 1
            dfact[()] = dfact[()] + len(w) + 1 if () in dfact else len(w) + 1
            for i in range(len(w)):
                dpref[w[:i + 1]] = dpref[w[:i + 1]] + 1 \
                    if w[:i + 1] in dpref else 1
                dsuff[w[i:]] = dsuff[w[i:]] + 1 if w[i:] in dsuff else 1
                for j in range(i + 1, len(w) + 1):
                    dfact[w[i:j]] = dfact[w[i:j]] + 1 if w[i:j] in dfact else 1
            line = f.readline()
        f.close()
        # self._create_pickle_files(adr=adr, dsample=dsample, dpref=dpref,
        #                           dsuff=dsuff, dfact=dfact)
        return nbL, nbEx, dsample, dpref, dsuff, dfact

    def _create_pickle_files(self, adr, dsample, dpref, dsuff, dfact):
        f = open(adr + ".sample.pkl", "wb")
        pickle.dump(dsample, f)
        f.close()
        f = open(adr + ".pref.pkl", "wb")
        pickle.dump(dpref, f)
        f.close()
        f = open(adr + ".suff.pkl", "wb")
        pickle.dump(dsuff, f)
        f.close()
        f = open(adr + ".fact.pkl", "wb")
        pickle.dump(dfact, f)
        f.close()
