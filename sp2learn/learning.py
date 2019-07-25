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
"""This module contains the Learning class

.. module author:: François Denis

"""

from __future__ import division, print_function
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as lin
import math
from sp2learn.sample import Sample
from sp2learn.hankel import Hankel
import sp2learn.automaton as AC

class Learning(object):
    """ A learning instance

    :Example:

    >>> from sp2learn import Learning, Sample
    >>> train_file = '0.spice.train'
    >>> pT = Sample(adr=train_file)
    >>> S_app = Learning(sample_instance=pT)

    - Input:

    :param Sample sample_instance: a an instance of Sample
    (nbL, nbEx, and dictionaries )

    """

    def __init__(self, sample_instance):

        # # Size of the alphabet
        # self.nbL = sample_instance.nbL
        # # Number of samples
        # self.nbEx = sample_instance.nbEx
        # # The dictionary that contains the sample
        # self.sample = sample_instance.sample
        # # The dictionary that contains the prefixes
        # self.pref = sample_instance.pref
        # # The dictionary that contains the suffixes
        # self.suff = sample_instance.suff
        # # The dictionary that contains the factors
        # self.fact = sample_instance.fact
        # The Sample object that contains Samples dictionaries
        self.sample_object = sample_instance

    @property
    def sample_object(self):
        """Sample object, contains dictionaries"""

        return self._sample_object

    @sample_object.setter
    def sample_object(self, sample_object):
         if not isinstance(sample_object, Sample):
             raise TypeError("sample_object should be a Sample")
         self._sample_object = sample_object
    #
    # @property
    # def nbL(self):
    #     """Size of the alphabet"""
    #     return self._nbL
    #
    # @nbL.setter
    # def nbL(self, nbL):
    #     if not isinstance(nbL, int):
    #         raise TypeError("nbL should be an integer")
    #     if nbL < 0:
    #         raise ValueError("The size of the alphabet should " +
    #                          "an integer >= 0")
    #     self._nbL = nbL
    #
    # @property
    # def nbEx(self):
    #     """Number of examples"""
    #     return self._nbEx
    #
    # @nbEx.setter
    # def nbEx(self, nbEx):
    #     if not isinstance(nbEx, int):
    #         raise TypeError("nbEx should be an integer")
    #     if nbEx < 0:
    #         raise ValueError("The number of examples should be " +
    #                          " an integer >= 0")
    #     self._nbEx = nbEx
    #
    # @property
    # def sample(self):
    #     """The dictionary that contains the sample """
    #     return self._sample
    #
    # @sample.setter
    # def sample(self, sample):
    #     if isinstance(sample, dict):
    #         self._sample = sample
    #     else:
    #         raise TypeError("sample should be a dictionary.")
    #
    # @property
    # def pref(self):
    #     """The dictionary that contains the prefixes"""
    #     return self._pref
    #
    # @pref.setter
    # def pref(self, pref):
    #     if isinstance(pref, dict):
    #         self._pref = pref
    #     else:
    #         raise TypeError("pref should be a dictionary.")
    #
    # @property
    # def suff(self):
    #     """The dictionary that contains the suffixes"""
    #     return self._suff
    #
    # @suff.setter
    # def suff(self, suff):
    #     if isinstance(suff, dict):
    #         self._suff = suff
    #     else:
    #         raise TypeError("suff should be a dictionary.")
    #
    # @property
    # def fact(self):
    #     """The dictionary that contains the factors"""
    #     return self._fact
    #
    # @fact.setter
    # def fact(self, fact):
    #     if isinstance(fact, dict):
    #         self._fact = fact
    #     else:
    #         raise TypeError("fact should be a dictionary.")
    #
    #
    @staticmethod
    def BuildAutomatonFromHankel(lhankel, nbL, rank, sparse=False):
        """ Build an automaton from Hankel matrix

        - Input:

        :param list lhankel: list of Hankel matrix
        :param int nbL: the number of letters
        :param int rank: the ranking number
        :param boolean sparse: (default value = False) True if Hankel
               matrix is sparse

        - Output:

        :returns: An automaton instance
        :rtype: Automaton
        """

        print ("Start Building Automaton from Hankel matrix")
        if not sparse:
            hankel = lhankel[0]
            [u, s, v] = np.linalg.svd(hankel)
            u = u[:, :rank]
            v = v[:rank, :]
            # ds = np.zeros((rank, rank), dtype=complex)
            ds = np.diag(s[:rank])
            pis = np.linalg.pinv(v)
            del v
            pip = np.linalg.pinv(np.dot(u, ds))
            del u, ds
            init = np.dot(hankel[0, :], pis)
            term = np.dot(pip, hankel[:, 0])
            trans = []
            for x in range(nbL):
                hankel = lhankel[x+1]
                trans.append(np.dot(pip, np.dot(hankel, pis)))

        else:
            hankel = lhankel[0]
            [u, s, v] = lin.svds(hankel, k=rank)
            ds = np.diag(s)
            pis = np.linalg.pinv(v)
            del v
            pip = np.linalg.pinv(np.dot(u, ds))
            del u, ds
            init = hankel[0, :].dot(pis)[0, :]
            term = np.dot(pip, hankel[:, 0].toarray())[:, 0]
            trans = []
            for x in range(nbL):
                hankel = lhankel[x+1]
                trans.append(np.dot(pip, hankel.dot(pis)))
        A = AC.Automaton(nbL, rank, init, term, trans)
        # ms=np.linalg.pinv(vt)
        # init=h.dot(ms)/self.nbEx
        # print(type(init),init.shape)
        # mp=np.linalg.pinv(u.dot(np.diag(s)))
        # v=v.todense()
        # term=np.dot(mp,v)
        # trans=[] # a suivre
        print ("End of Automaton computation")
        return A

    def LearnAutomaton(self, rank, lrows=[], lcolumns=[], version="classic",
                       partial=False, sparse=False):
        """ Learn Automaton from sample

        - Input:

        :param int rank: the ranking number

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
        :param boolean partial: (default value = False) build
               of partial Hankel matrix
        :param boolean sparse: (default value = False) True if Hankel
               matrix is sparse

        - Output:

        :returns: An automaton instance
        :rtype: Automaton
        """

        lhankel = Hankel(sample_instance=self.sample_object,
                         lrows=lrows, lcolumns=lcolumns,
                         version=version,
                         partial=partial, sparse=sparse).lhankel
        matrix_shape =min(lhankel[0].shape)
        if (min(lhankel[0].shape) < rank) :
            raise ValueError("The number of rank "+  str(rank)
                             + "should be <= to " +
                              "Hankel Matrix shape " + str(matrix_shape) )

        A = self.BuildAutomatonFromHankel(lhankel=lhankel,
                                          nbL=self.sample_object.nbL,
                                          rank=rank, sparse=sparse)

        A.initial = A.initial / self.sample_object.nbEx
        if version == "prefix":
            A = A.transformation(source="prefix", target="classic")
        if version == "factor":
            A = A.transformation(source="factor", target="classic")
        return A

    @staticmethod
    def Perplexity(A, adr):
        """ Perplexity calculation """
        Cible = AC.Automaton.load_Spice_Automaton("./" + adr + ".target")
        Test = Learning(adr="./"+adr+".test")
        sA, sC = 0, 0
        for w in Test.sample_object.sample:
            sA = sA + abs(A.val(w))
            sC = sC + abs(Cible.val(w))
        s = 0
        for w in Test.sample_object.sample:
            s = s + Cible.val(w)/sC*math.log(abs(A.val(w))/sA)
        p = math.exp(-s)
        return p

# if __name__ == '__main__':
#     from skgilearn.datasets.get_dataset_path import get_dataset_path
#     adr = get_dataset_path("essai")
#     P = Learning(adr=adr, type='SPiCe')
#
#     print("nbL = " + str(P.nbL))
#     print("nbEx = " + str(P.nbEx))
#     print("samples = " + str(P.sample))
#     print("prefixes = " + str(P.pref))
#     print("suffixes = " + str(P.suff))
#     print("factors = " + str(P.fact))
