import numpy as np
import sympy
from PIL import Image, ImageChops
from utils import execute as Exec
from utils import free_const
import io
import matplotlib.pyplot as plt
from configs import config

def DEFAULT_WRAPPER(func, X):
    return func(X)

class Program:
    """
    Interface class representing a single program.
    Attributes
    ----------
    tokens : array_like of phytokens.Token
        Tokens making up program.
    size : int
        Size of program.
    library : library.Library
        Library of phytokens that could appear in Program.
    is_physical : bool or None
        Is program physical (units-wize) ?
    free_const_values : array_like of float or None
        Values of free constants for each constant in the library.
    is_opti : numpy.array of shape (1,) of bool or None
        Is set of free constant optimized ? Use is_opti[0] to access the value.
    opti_steps : numpy.array of shape (1,) of int or None
        Number of iterations necessary to optimize the set of free constants. Use opti_steps[0] to access the value.
    candidate_wrapper : callable
        Wrapper to apply to candidate program's output, candidate_wrapper taking func, X as arguments where func is
        a candidate program callable (taking X as arg). By default = None, no wrapper is applied (identity).
    """
    def __init__(self, tokens, library, is_physical = None, free_const_values = None, is_opti = None, opti_steps = None, candidate_wrapper = None):
        """
        Parameters
        ----------
        See attributes help for details.
        """
        # Asserting that phytokens make up a full tree representation, no more, no less
        total_arity = np.sum([tok.arity for tok in tokens])
        assert len(tokens)-total_arity==1, "Tokens making up Program must consist in a full tree representation " \
                                           "(length - total arities = 1), no more, no less"
        self.tokens       = tokens
        self.size         = len(tokens)
        self.library      = library
        self.is_physical  = is_physical

        if candidate_wrapper is None:
            candidate_wrapper = DEFAULT_WRAPPER
        self.candidate_wrapper = candidate_wrapper

        # ----- free const related -----
        # Values
        self.free_const_values = free_const_values                                                  # (?,)
        # Using an array of shape (1,) (ie. reference) in order to affect the underlying values in the
        # FreeConstantsTable object.
        self.is_opti           = is_opti                                                            # (1,)
        self.opti_steps        = opti_steps                                                         # (1,)

    def execute_wo_wrapper(self, X):
        """
        Executes program on X.
        Parameters
        ----------
        X : torch.tensor of shape (n_dim, ?,) of float
            Values of the input variables of the problem with n_dim = nb of input variables.
        Returns
        -------
        y : torch.tensor of shape (?,) of float
            Result of computation.
        """
        y = Exec.ExecuteProgram(input_var_data     = X,
                                 free_const_values = self.free_const_values,
                                 program_tokens    = self.tokens)
        return y

    def execute(self, X):
        """
        Executes program on X.
        Parameters
        ----------
        X : torch.tensor of shape (n_dim, ?,) of float
            Values of the input variables of the problem with n_dim = nb of input variables.
        Returns
        -------
        y : torch.tensor of shape (?,) of float
            Result of computation.
        """
        y = self.candidate_wrapper(lambda X: self.execute_wo_wrapper(X), X)
        return y

    def optimize_constants(self, X, y_target, args_opti = None):
        """
        Optimizes free constants of program.
        Parameters
        ----------
        X : torch.tensor of shape (n_dim, ?,) of float
            Values of the input variables of the problem with n_dim = nb of input variables.
        y_target : torch.tensor of shape (?,) of float
            Values of target output.
        args_opti : dict or None, optional
            Arguments to pass to free_const.optimize_free_const. By default, free_const.DEFAULT_OPTI_ARGS
            arguments are used.
        """
        if args_opti is None:
            args_opti = free_const.DEFAULT_OPTI_ARGS
        func_params = lambda params: self.__call__(X)

        history = free_const.optimize_free_const (     func     = func_params,
                                                       params   = self.free_const_values,
                                                       y_target = y_target,
                                                       **args_opti)

        # Logging optimization process
        self.is_opti    [0] = True
        self.opti_steps [0] = len(history)  # Number of iterations it took to optimize the constants

        return history

    def __call__(self, X):
        """
        Executes program on X.
        """
        return self.execute(X=X)

    def __getitem__(self, key):
        """
        Returns phytokens at position = key.
        """
        return self.tokens[key]

    def __repr__(self):
        return str(self.tokens)

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------- REPRESENTATION : INFIX RELATED -----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def get_infix_str (self):
        """
        Computes infix str representation of a program.
        (which is the usual way to note symbolic function: +34 (in polish notation) = 3+4 (in infix notation))
        Returns
        -------
        program_str : str
        """
        program_str = Exec.ComputeInfixNotation(self.tokens)
        return program_str

    def get_infix_sympy (self, do_simplify = False):
        """
        Returns sympy symbolic representation of a program.
        Parameters
        ----------
        do_simplify : bool
            If True performs a symbolic simplification of program.
        Returns
        -------
        program_sympy : sympy.core
            Sympy symbolic function. It is possible to run program_sympy.evalf(subs={'x': 2.4}) where 'x' is a variable
            appearing in the program to evaluate the function with x = 2.4.
        """
        program_str = self.get_infix_str()
        program_sympy = sympy.parsing.sympy_parser.parse_expr(program_str, evaluate=False)
        if do_simplify:
            program_sympy = sympy.simplify(program_sympy, rational=True) # 2.0 -> 2
        return program_sympy

    def get_infix_pretty (self, do_simplify = False):
        """
        Returns a printable ASCII sympy.pretty representation of a program.
        Parameters
        ----------
        do_simplify : bool
            If True performs a symbolic simplification of program.
        Returns
        -------
        program_pretty_str : str
        """
        program_sympy = self.get_infix_sympy(do_simplify = do_simplify)
        program_pretty_str = sympy.pretty (program_sympy)
        return program_pretty_str


    def get_infix_latex (self,replace_dummy_symbol = True, new_dummy_symbol = "?", do_simplify = True):
        """
        Returns an str latex representation of a program.
        Parameters
        ----------
        replace_dummy_symbol : bool
            If True, dummy symbol is replaced by new_dummy_symbol.
        new_dummy_symbol : str or None
            Replaces dummy symbol if replace_dummy_symbol is True.
        do_simplify : bool
            If True performs a symbolic simplification of program.
        Returns
        -------
        program_latex_str : str
        """
        program_sympy = self.get_infix_sympy(do_simplify=do_simplify)
        program_latex_str = sympy.latex (program_sympy)
        if replace_dummy_symbol:
            program_latex_str = program_latex_str.replace(config.positional_token_default_values['DUMMY_TOKEN_NAME'], new_dummy_symbol)
        return program_latex_str


    def get_infix_fig (self,
                       replace_dummy_symbol = True,
                       new_dummy_symbol = "?",
                       do_simplify = True,
                       show_superparent_at_beginning = True,
                       text_size = 16,
                       text_pos  = (0.0, 0.5),
                       figsize   = (10, 2),
                       ):
        """
        Returns pyplot (figure, axis) containing analytic symbolic function program.
        Parameters
        ----------
        replace_dummy_symbol : bool
            If True, dummy symbol is replaced by new_dummy_symbol.
        new_dummy_symbol : str or None
            Replaces dummy symbol if replace_dummy_symbol is True.
        do_simplify : bool
            If True performs a symbolic simplification of program.
        show_superparent_at_beginning : bool
            If True, shows superparent in Figure like "y = ..." instead of just "..."
        text_size : int
            Size of text in figure.
        text_pos : (float, float)
            Position of text in figure.
        figsize : (int, int)
            Shape of figure.
        Returns
        -------
        fig, ax : matplotlib.pyplot.Figure, matplotlib.pyplot.AxesSubplot
        """
        # Latex str of symbolic function
        latex_str = self.get_infix_latex(replace_dummy_symbol = replace_dummy_symbol,
                                         new_dummy_symbol = new_dummy_symbol,
                                         do_simplify = do_simplify)
        # Adding "superparent =" before program to make it pretty
        if show_superparent_at_beginning:
            latex_str = self.library.superparent.name + ' =' + latex_str
        # Fig
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        # enables new_dummy_symbol = "\square"
        plt.rc('text.latex', preamble=r'\usepackage{amssymb} \usepackage{xcolor}')
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.axis('off')
        ax.text(text_pos[0], text_pos[1], f'${latex_str}$', size = text_size)
        return fig, ax


    def get_infix_image(self,
                        replace_dummy_symbol = True,
                        new_dummy_symbol = "?",
                        do_simplify = True,
                        text_size    = 16,
                        text_pos     = (0.0, 0.5),
                        figsize      = (8, 2),
                        dpi          = 512,
                        fpath        = None,
                        ):
        """
        Returns image containing analytic symbolic function program.
        Parameters
        ----------
        replace_dummy_symbol : bool
            If True, dummy symbol is replaced by new_dummy_symbol.
        new_dummy_symbol : str or None
            Replaces dummy symbol if replace_dummy_symbol is True.
        do_simplify : bool
            If True performs a symbolic simplification of program.
        text_size : int
            Size of text in figure.
        text_pos : (float, float)
            Position of text in figure.
        figsize : (int, int)
            Shape of figure.
        dpi : int
            Pixel density for raster image.
        fpath : str or None
            Path where to save image. Default = None, not saved.
        Returns
        -------
        image : PIL.Image.Image
        """
        # Getting fig, ax
        fig, ax = self.get_infix_fig (
                            replace_dummy_symbol = replace_dummy_symbol,
                            new_dummy_symbol = new_dummy_symbol,
                            do_simplify = do_simplify,
                            text_size = text_size,
                            text_pos  = text_pos,
                            figsize   = figsize,
                            )

        # Exporting image to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi)
        plt.close()

        # Buffer -> img
        white = (255, 255, 255, 255)
        img = Image.open(buf)
        bg = Image.new(img.mode, img.size, white)
        diff = ImageChops.difference(img, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        img = img.crop(bbox)

        # Saving if fpath is given
        if fpath is not None:
            fig.savefig(fpath, dpi=dpi)

        return img

    def show_infix(self,
                   replace_dummy_symbol = True,
                   new_dummy_symbol = "?",
                   do_simplify = False,
                   text_size=24,
                   text_pos=(0.0, 0.5),
                   figsize=(10, 1),
                   ):
        """
        Shows pyplot (figure, axis) containing analytic symbolic function program.
        Parameters
        ----------
        replace_dummy_symbol : bool
            If True, dummy symbol is replaced by new_dummy_symbol.
        new_dummy_symbol : str or None
            Replaces dummy symbol if replace_dummy_symbol is True.
        do_simplify : bool
            If True performs a symbolic simplification of program.
        text_size : int
            Size of text in figure.
        text_pos : (float, float)
            Position of text in figure.
        figsize : (int, int)
            Shape of figure.
        """
        # Getting fig, ax
        fig, ax = self.get_infix_fig (
                            replace_dummy_symbol = replace_dummy_symbol,
                            new_dummy_symbol = new_dummy_symbol,
                            do_simplify = do_simplify,
                            text_size = text_size,
                            text_pos  = text_pos,
                            figsize   = figsize,
                            )
        # Show
        plt.show()
        return None
