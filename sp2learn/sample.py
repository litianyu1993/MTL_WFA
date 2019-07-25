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
"""This module contains the Sample class
The Sample class encapsulates a sample 's components
nbL and nbEx numbers, the fourth dictionaries for sample, prefix,
suffix and factor

"""

from sp2learn.load import Load


class Sample(object):
    """ A sample instance

    :Example:

    >>> from sp2learn import Sample
    >>> train_file = '0.spice.train'
    >>> pT = Sample(adr=train_file)

    - Input:

    :param string adr: adresse and name of the loaden file
    :param string type: (default value = 'SPiCe') indicate
           the structure of the file
    :param lrows: number or list of rows,
           a list of strings if partial=True;
           otherwise, based on self.pref if version="classic" or
           "prefix", self.fact otherwise
    :type lrows: int or list of int
    :param lcolumns: number or list of columns
           a list of strings if partial=True ;
           otherwise, based on self.suff if version="classic" or "suffix",
           self.fact otherwise
    :type lcolumns: int or list of int
    :param string version: (default = "classic") version name
    :param boolean partial: (default value = False) build of partia

    """

    def __init__(self, adr, type='SPiCe', lrows=[], lcolumns=[],
                 version ="classic", partial=False):

        # Size of the alphabet
        self._nbL = None
        # Number of samples
        self._nbEx = None
        # The dictionary that contains the sample
        self._sample = {}
        # The dictionary that contains the prefixes
        self._pref = {}
        # The dictionary that contains the suffixes
        self._suff = {}
        # The dictionary that contains the factors
        self._fact = {}
        if type == 'SPiCe':
            l = Load(adr).load_Spice_Sample(
                lrows=lrows, lcolumns=lcolumns,
                version=version, partial=partial)
            self._nbL = l[0]
            self._nbEx = l[1]
            self._sample = l[2]
            self._pref = l[3]
            self._suff = l[4]
            self._fact = l[5]

    @property
    def nbL(self):
        """Number of letters"""

        return self._nbL

    @nbL.setter
    def nbL(self, nbL):
        if not isinstance(nbL, int):
            raise TypeError("nbL should be an integer")
        if nbL < 0:
            raise ValueError("The size of the alphabet should " +
                             "an integer >= 0")
        self._nbL = nbL

    @property
    def nbEx(self):
        """Number of examples"""

        return self._nbEx

    @nbEx.setter
    def nbEx(self, nbEx):
        if not isinstance(nbEx, int):
            raise TypeError("nbEx should be an integer")
        if nbEx < 0:
            raise ValueError("The number of examples should be " +
                             " an integer >= 0")
        self._nbEx = nbEx

    @property
    def sample(self):
        """sample dictionary"""

        return self._sample

    @sample.setter
    def sample(self, sample):
        if isinstance(sample, dict):
            self._sample = sample
        else:
            raise TypeError("sample should be a dictionnary.")

    @property
    def pref(self):
        """prefix dictionary"""

        return self._pref

    @pref.setter
    def pref(self, pref):
        if isinstance(pref, dict):
            self._pref = pref
        else:
            raise TypeError("pref should be a dictionnary.")

    @property
    def suff(self):
        """suffix dictionary"""

        return self._suff

    @suff.setter
    def suff(self, suff):
        if isinstance(suff, dict):
            self._suff = suff
        else:
            raise TypeError("suff should be a dictionnary.")

    @property
    def fact(self):
        """factor dictionary"""

        return self._fact

    @fact.setter
    def fact(self, fact):
        if isinstance(fact, dict):
            self._fact = fact
        else:
            raise TypeError("fact should be a dictionnary.")

    def select_rows(self, nb_rows_max=1000, version='classic'):
        """define lrows

        - Input:

        :param int nb_rows_max:  (default = 1000) number of maximum rows
        :param string version: (default = "classic") version name

        - Output:

        :param list lrows:  list of rows

        """

        lRows = [] # liste à renvoyer
        lLeafs = [([],self.suff[()])]  # la liste de couples (prefixes frontières, nb occ) initialisée au prefixe vide
        nbRows = 0
        if version == 'classic':
            while lLeafs and nbRows < nb_rows_max:
                lastWord = lLeafs.pop()[0] # le prefixe frontière le plus fréquent
                lRows.append(tuple(lastWord))
                nbRows += 1
                for i in range(self.nbL):
                    newWord = lastWord + [i] # successeur de lastword
                    tnewWord = tuple(newWord) # tuple associé
                    if tnewWord in self.pref:
                        # ajout d'un nouveau prefixe frontière
                        lLeafs.append((newWord, self.pref[tnewWord]))
                lLeafs = sorted(lLeafs, key = lambda x: x[1])
        elif version == 'prefix':
            while lLeafs and nbRows < nb_rows_max:
                lastWord = lLeafs.pop()[0] # le prefixe frontière le plus fréquent
                lRows.append(tuple(lastWord))
                nbRows += 1
                for i in range(self.nbL):
                    newWord = lastWord + [i] # successeur de lastword
                    tnewWord = tuple(newWord) # tuple associé
                    if tnewWord in self.pref:
                        # ajout d'un nouveau prefixe frontière
                        nb = 0
                        for u in self.sample:
                            if tnewWord <= u:
                                nb += self.sample[u]*(len(u) - len(tnewWord) + 1)
                        lLeafs.append((newWord, nb))
                lLeafs = sorted(lLeafs, key = lambda x: x[1])
        elif version == 'factor':
            while lLeafs and nbRows < nb_rows_max:
                lastWord = lLeafs.pop()[0] # le prefixe frontière le plus fréquent
                lRows.append(tuple(lastWord))
                nbRows += 1
                for i in range(self.nbL):
                    newWord = lastWord + [i] # successeur de lastword
                    tnewWord = tuple(newWord) # tuple associé
                    if tnewWord in self.fact:
                        # ajout d'un nouveau prefixe frontière
                        nb = 0
                        lw = len(tnewWord)
                        for u in self.sample:
                            if len(u) >= lw:
                                for i in range(lw,len(u)+1):
                                    if u[:i][-lw:] == tnewWord:
                                        nb += self.sample[u]*(len(u) - i + 1)
                        lLeafs.append((newWord, nb))
                lLeafs = sorted(lLeafs, key = lambda x: x[1])
            #print(lLeafs)
        return lRows


    def select_columns(self, nb_columns_max=1000, version ='classic'):
        """define lcolumns

        - Input:

        :param int nb_columns_max:  (default = 1000) number of maximum columns
        :param string version: (default = "classic") version name

        - Output:

        :param list lcolumns:  list of columns

        """

        lColumns = [] # liste à renvoyer
        lLeafs = [([],self.suff[()])]  # la liste de couples (suffixes frontières, nb occ) initialisée au suffixe vide
        nbColumns = 0
        if version == 'classic':
            while lLeafs and nbColumns < nb_columns_max:
                lastWord = lLeafs.pop()[0] # le suffixe frontière le plus fréquent
                lColumns.append(tuple(lastWord))
                nbColumns += 1
                for i in range(self.nbL):
                    newWord = lastWord + [i] # successeur de lastword
                    tnewWord = tuple(newWord) # tuple associé
                    if tnewWord in self.suff:
                        # ajout d'un nouveau suffixe frontière
                        lLeafs.append((newWord, self.suff[tnewWord]))
                lLeafs = sorted(lLeafs, key = lambda x: x[1]) # suffixe le plus fréquent en dernier
            #print(lLeafs)
        elif version == 'prefix':
            while lLeafs and nbColumns < nb_columns_max:
                lastWord = lLeafs.pop()[0] # le prefixe frontière le plus fréquent
                lColumns.append(tuple(lastWord))
                nbColumns += 1
                for i in range(self.nbL):
                    newWord = lastWord + [i] # successeur de lastword
                    tnewWord = tuple(newWord) # tuple associé
                    if tnewWord in self.fact:
                        # ajout d'un nouveau suffixe frontière
                        lLeafs.append((newWord, self.fact[tnewWord]))
                lLeafs = sorted(lLeafs, key = lambda x: x[1])
        elif version == 'factor':
            while lLeafs and nbColumns < nb_columns_max:
                lastWord = lLeafs.pop()[0] # le prefixe frontière le plus fréquent
                lColumns.append(tuple(lastWord))
                nbColumns += 1
                for i in range(self.nbL):
                    newWord = lastWord + [i] # successeur de lastword
                    tnewWord = tuple(newWord) # tuple associé
                    if tnewWord in self.fact:
                        # ajout d'un nouveau prefixe frontière
                        nb = 0
                        lw = len(tnewWord)
                        for u in self.sample:
                            if len(u) >= lw:
                                for i in range(lw,len(u)+1):
                                    if u[:i][-lw:] == tnewWord:
                                        nb += self.sample[u]*(i - lw + 1)
                        lLeafs.append((newWord, nb))
                lLeafs = sorted(lLeafs, key = lambda x: x[1])
            #print(lLeafs)
        return lColumns
