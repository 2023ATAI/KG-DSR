import numpy as np
import copy as copy

class Cursor:
    """
    Helper class for single-phytokens navigation in tree of programs in VectPrograms.
    Represents the position of a single phytokens in a program in a batch of programs.
    For user-exploration, program testing and debugging.
    Attributes
    ----------
    programs : program.VectPrograms
        Batch of programs to explore.
    prog_idx : int
        Initial position of cursor in batch dim (= index of program in batch).
    pos : int
        Initial position of cursor in time dim (= index of phytokens in program).
    Methods
    -------
    coords () -> numpy.array of shape (2, 1) of int
        Returns current coordinates in batch (batch dim, time dim) compatible with VectPrograms methods.
    set_pos (new_pos : int) -> program.Cursor
        Sets position of cursor in time dim (= index of phytokens in program) and returns cursor.
    child   (i_child   : int) -> program.Cursor
        Returns a cursor pointing to child number i_child of current phytokens. Raises error if there is no child.
    sibling (i_sibling : int) -> program.Cursor
        Returns a cursor pointing to sibling number i_sibling of current phytokens. Raises error if there is no sibling .
        cursor.
    parent () -> program.Cursor
        Returns a cursor pointing to parent of current phytokens. Raises error if there is no parent.
    """
    def __init__(self, programs, prog_idx=0, pos=0):
        """
        See class documentation.
        Parameters
        ----------
        programs : program.VectPrograms
        prog_idx : int
        pos : int
        """
        self.programs = programs
        self.prog_idx = prog_idx
        self.pos      = pos

    @property
    def coords(self):
        """
        See class documentation.
        Returns
        -------
        coords : numpy.array of shape (2, 1) of int
        """
        return np.array([[self.prog_idx], [self.pos]])

    @property
    def token(self):
        """
        Returns phytokens object at coords pointed by cursor.
        Returns
        -------
        phytokens : token.Token
        """
        return self.programs.get_token(self.coords)[0]

    def token_prop (self, attr):
        """
        Returns attr attribute in VectPrograms of the phytokens at coords pointed by cursor.
        Returns
        -------
        token_prop : ?
            ? depends on the property.
        """
        return getattr(self.programs.tokens, attr)[tuple(self.coords)][0]

    def set_pos(self, new_pos = 0):
        """
        See class documentation.
        Parameters
        ----------
        new_pos : int
        Returns
        -------
        self : program.Cursor
        """
        self.pos = new_pos
        return self

    def child(self, i_child = 0):
        """
        See class documentation.
        Parameters
        ----------
        i_child : int
        Returns
        -------
        self : program.Cursor
        """
        has_relative     = self.programs.tokens.has_children_mask[tuple(self.coords)][0]
        if not has_relative:
            err_msg = "Unable to navigate to child, Token %s at pos = %i (program %i) has no child." % (
            str(self), self.pos, self.prog_idx)
            raise IndexError(err_msg)
        pos_children    = self.programs.get_children(tuple(self.coords))[1:, 0]
        child = copy.deepcopy(self)
        child.pos        = pos_children[i_child]
        return child

    @property
    def sibling(self, i_sibling = 0):
        """
        See class documentation.
        Parameters
        ----------
        i_sibling : int
        Returns
        -------
        self : program.Cursor
        """
        has_relative = self.programs.tokens.has_siblings_mask[tuple(self.coords)][0]
        if not has_relative:
            err_msg = "Unable to navigate to sibling, Token %s at pos = %i (program %i) has no sibling." % (
                str(self), self.pos, self.prog_idx)
            raise IndexError(err_msg)
        pos_siblings = self.programs.get_siblings(tuple(self.coords))[1:, 0]
        sibling = copy.deepcopy(self)
        sibling.pos     = pos_siblings[i_sibling]
        return sibling

    @property
    def parent(self,):
        """
        See class documentation.
        Returns
        -------
        self : program.Cursor
        """
        has_relative = self.programs.tokens.has_parent_mask[tuple(self.coords)][0]
        if not has_relative:
            err_msg = "Unable to navigate to parent, Token %s at pos = %i (program %i) has no parent." % (
                str(self), self.pos, self.prog_idx)
            raise IndexError(err_msg)
        pos_parent = self.programs.get_parent(tuple(self.coords))[1, 0]
        parent = copy.deepcopy(self)
        parent.pos   = pos_parent
        return parent

    def __repr__(self):
        return self.programs.lib_names[self.programs.tokens.idx[tuple(self.coords)]][0]
