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
"""This module contains the Hankel class

"""
from __future__ import division, print_function
import scipy.sparse as sps
import numpy as np


class Hankel(object):
    """ A Hankel instance , compute the list of Hankel matrix

    :Example:

    >>> from sp2learn import Sample, Learning, Hankel
    >>> train_file = '0.spice.train'
    >>> pT = Sample(adr=train_file)
    >>> S_app = Learning(sample_instance=pT)
    >>> lhankel = Hankel( sample=pT.sample, pref=pT.pref,
    >>>                   suff=pT.suff, fact=pT.fact,
    >>>                   nbL=pT.nbL, nbEx=pT.nbEx,
    >>>                   lrows=6, lcolumns=6, version="classic",
    >>>                   partial=True, sparse=True).lhankel

    - Input:

    :param dict sample: sample dictionary
    :param dict pref: prefix dictionary
    :param dict suff: suffix dictionary
    :param dict fact: factor dictionary
    :param int nbL: the number of letters
    :param int nbS: the number of states
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
    :param boolean partial: (default value = False) build of partial
    :param boolean sparse: (default value = False) True if Hankel
           matrix is sparse
    """

    def __init__(
            self, sample_instance,
            lrows=[], lcolumns=[],
            version="classic", partial=False, sparse=False ):
        # Size of the alphabet
        self.nbL = sample_instance.nbL
        # Number of samples
        self.nbEx = sample_instance.nbEx
        self.version = version
        self.partial = partial
        self.sparse = sparse
        self.lhankel = self.build(sample=sample_instance.sample,
                                  pref=sample_instance.pref,
                                  suff=sample_instance.suff,
                                  fact=sample_instance.fact,
                                  lrows=lrows, lcolumns=lcolumns)

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

    def build(self, sample, pref, suff, fact, lrows, lcolumns):

        """ Create a Hankel matrix

        - Input:

        :param dict sample: sample dictionary
        :param dict pref: prefix dictionary
        :param dict suff: suffix dictionary
        :param dict fact: factor dictionary
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

        - Output:

        :returns: list lhankel, list of hankel matrix,
                  a DoK based sparse matrix or nuppy matrix based not sparse
        :rtype: list of matrix

        """
        # calcul des lignes lrows et colonnes lcolumns
        if not self.partial:
            (lrows, lcolumns) = self._build_not_partial(
                pref=pref, suff=suff, fact=fact)
        else:
            (lrows, lcolumns) = self._build_partial(
                pref=pref, suff=suff, fact=fact,
                lrows=lrows, lcolumns=lcolumns)

        lhankel = self._create_hankel(sample=sample, pref=pref,
                                      suff=suff, fact=fact,
                                      lrows=lrows, lcolumns=lcolumns)
        return lhankel

    def _build_not_partial(self,pref, suff, fact):
        version = self.version
        if version == "classic":
            lrows = pref.keys()
            lcolumns = suff.keys()
        elif version == "prefix":
            lrows = pref.keys()
            lcolumns = fact.keys()
        elif version == "suffix":
            lrows = fact.keys()
            lcolumns = suff.keys()
        else:
            lrows = fact.keys()
            lcolumns = fact.keys()

        return (lrows, lcolumns)

    def _build_partial(self,
                       pref, suff, fact,
                       lrows, lcolumns):
        version = self.version

        if version == "classic":
            (lrows, lcolumns) = self._construc_partial_lrows_lcolumns(
                dict_first=pref, dict_second=suff,
                lrows=lrows, lcolumns=lcolumns)
        elif version == "prefix":
            (lrows, lcolumns) = self._construc_partial_lrows_lcolumns(
                dict_first=pref, dict_second=fact,
                lrows=lrows, lcolumns=lcolumns)
        elif version == "suffix":
            (lrows, lcolumns) = self._construc_partial_lrows_lcolumns(
                dict_first=fact, dict_second=suff,
                lrows=lrows, lcolumns=lcolumns)
        else:
            (lrows, lcolumns) = self._construc_partial_lrows_lcolumns(
                dict_first=fact, dict_second=fact,
                lrows=lrows, lcolumns=lcolumns)
        return lrows, lcolumns

    def _construc_partial_lrows_lcolumns(self, dict_first, dict_second,
                                         lrows,
                                         lcolumns):

        if isinstance(lrows, int):
            longmax = lrows
            lrows = [w for w in dict_first if len(w) <= longmax]
        else:
            s_first = set(dict_first)  # corresponding set
            lrows = [w for w in lrows if w in s_first]
        if isinstance(lcolumns, int):
            longmax = lcolumns
            lcolumns = [w for w in dict_second if len(w) <= longmax]
        else:
            s_second = set(dict_second)  # corresponding set
            lcolumns = [w for w in lcolumns if w in s_second]
        return (lrows, lcolumns)

    def _create_hankel(self, sample, pref, suff, fact, lrows, lcolumns):
        version = self.version
        sparse = self.sparse

        (drows, dcolumns) = self._sorted_rows_columns(lrows, lcolumns)

        nbRows = len(lrows)
        nbColumns = len(lcolumns)
        srows = set(lrows)
        scolumns = set(lcolumns)

        if sparse:
            lhankel = [sps.dok_matrix((nbRows, nbColumns)) for
                       i in range(self.nbL+1)]
        else:
            lhankel = [np.zeros((nbRows, nbColumns)) for
                       k in range(self.nbL+1)]
        if version == "classic":
            dsample = sample
        elif version == "prefix":
            dsample = pref
        elif version == "suffix":
            dsample = suff
        else:
            dsample = fact
        for w in dsample:
            for i in range(len(w)+1):
                if w[:i] in srows:
                    if w[i:] in scolumns:
                        lhankel[0][drows[w[:i]], dcolumns[w[i:]]] = dsample[w]
                    if (i < len(w) and w[i+1:] in scolumns):
                        lhankel[w[i]+1][drows[w[:i]],
                                        dcolumns[w[i+1:]]] = dsample[w]
        # for h in lhankel:
        #     print("h")
        #     print(h)
        # print(spsl.svds(hankel,k=6,return_singular_vectors=False))
        return lhankel

    def _sorted_rows_columns(self, lrows, lcolumns):
        nbRows = len(lrows)
        nbColumns = len(lcolumns)
        lrows = sorted(lrows, key=lambda x: (len(x), x))
        drows = {lrows[i]: i for i in range(nbRows)}
        lcolumns = sorted(lcolumns, key=lambda x: (len(x), x))
        dcolumns = {lcolumns[i]: i for i in range(nbColumns)}

        return (drows, dcolumns)
