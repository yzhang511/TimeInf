"""
Define some helper functions and Potts(), which inherits from AbstractModel.
"""

# standard libaries
import autograd
import autograd.numpy as np
from autograd.scipy.special import logsumexp
from scipy.special import gammaln
import time
import matplotlib.pyplot as plt
from itertools import product
from pgmpy.extern import tabulate
import random
import networkx as nx
from collections import defaultdict

# our implementation 
from structij.models.abstract_model import AbstractModel


# Helpers -----------------------------------------------------------------------
# factor class to help with partition computation. Adapted from
# https://github.com/pgmpy/pgmpy/blob/dev/pgmpy/factors/discrete/DiscreteFactor.py
class Factor:
    def __init__(self, variables, cardinality, log_values):
        """
        Inputs:
            variables: list, variable names
            cardinality: list, how many discrete values does each 
                variable take
            log_values: list, log-potential for each combination of variables,
                the latter variables' values changing faster
        """
        log_values = np.array(log_values, dtype=float)
        self.variables = list(variables)
        self.cardinality = np.array(cardinality, dtype=int)
        self.log_values = log_values.reshape(self.cardinality)
        return

    def scope(self):
        return self.variables

    def get_cardinality(self, variables):
        return {var: self.cardinality[self.variables.index(var)] for var in variables}

    def copy(self):
        """
        Returns a copy of the factor.
        """
        # not creating a new copy of self.values and self.cardinality
        # because __init__ methods does that.
        return Factor(
            self.scope(),
            self.cardinality,
            self.log_values
        )

    def product(self, phi1):
        """
        Return product factor between self and phi1. This routine
        is repeated many times to compute the potential over the 
        whole graph.
        """
        phi = self.copy()
        phi1 = phi1.copy()

        # modifying phi to add new variables
        extra_vars = set(phi1.variables) - set(phi.variables)
        if extra_vars:
            slice_ = [slice(None)] * len(phi.variables)
            slice_.extend([np.newaxis] * len(extra_vars))
            phi.log_values = phi.log_values[tuple(slice_)]

            phi.variables.extend(extra_vars)

            new_var_card = phi1.get_cardinality(extra_vars)
            phi.cardinality = np.append(
                phi.cardinality, [new_var_card[var] for var in extra_vars]
            )

        # modifying phi1 to add new variables
        extra_vars = set(phi.variables) - set(phi1.variables)
        if extra_vars:
            slice_ = [slice(None)] * len(phi1.variables)
            slice_.extend([np.newaxis] * len(extra_vars))
            phi1.log_values = phi1.log_values[tuple(slice_)]

            phi1.variables.extend(extra_vars)
            # No need to modify cardinality as we don't need it.

        # rearranging the axes of phi1 to match phi
        for axis in range(phi.log_values.ndim):
            exchange_index = phi1.variables.index(phi.variables[axis])
            phi1.variables[axis], phi1.variables[exchange_index] = (
                phi1.variables[exchange_index],
                phi1.variables[axis],
            )
            phi1.log_values = phi1.log_values.swapaxes(axis, exchange_index)

        phi.log_values = phi.log_values + phi1.log_values

        return phi

    def many_products(self, factors):
        """
        Take a product between self and many factors, returning a new factor.
        Inputs: 
            factors: list of Factor 
        """
        if len(factors) == 0:
            return self.copy()
        else:
            newphi = self.product(factors[0])
            for i in range(1, len(factors)):
                newphi = newphi.product(factors[i])
            return newphi

    def marginalize(self, variables, inplace=False):
        """
        Modifies the factor with marginalized values.

        Parameters
        ----------
        variables: list, array-like
            List of variables over which to marginalize.

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor.

        Returns
        -------
        DiscreteFactor or None: if inplace=True (default) returns None
                        if inplace=False returns a new `DiscreteFactor` instance.
        """

        phi = self if inplace else self.copy()

        for var in variables:
            if var not in phi.variables:
                raise ValueError("{var} not in scope.".format(var=var))

        var_indexes = [phi.variables.index(var) for var in variables]

        index_to_keep = sorted(set(range(len(self.variables))) - set(var_indexes))
        phi.variables = [phi.variables[index] for index in index_to_keep]
        phi.cardinality = phi.cardinality[index_to_keep]

        phi.log_values = logsumexp(phi.log_values, axis=tuple(var_indexes))

        if not inplace:
            return phi

    def __str__(self):
        return self._str(phi_or_p="phi", tablefmt="grid")

    def _str(self, phi_or_p="phi", tablefmt="grid", print_state_names=True):
        """
        Generate the string from `__str__` method. Factors are printed with
        values rather than log-values.
        Parameters
        ----------
        phi_or_p: 'phi' | 'p'
                'phi': When used for Factors.
                  'p': When used for CPDs.
        print_state_names: boolean
                If True, the user defined state names are displayed.
        """
        string_header = list(map(str, self.scope()))
        string_header.append(
            "{phi_or_p}({variables})".format(
                phi_or_p=phi_or_p, variables=",".join(string_header)
            )
        )

        value_index = 0
        factor_table = []
        for prob in product(*[range(card) for card in self.cardinality]):
            prob_list = [
                "{s}_{d}".format(s=list(self.variables)[i], d=prob[i])
                for i in range(len(self.variables))
            ]

            prob_list.append(np.exp(self.log_values.ravel()[value_index]))
            factor_table.append(prob_list)
            value_index += 1

        return tabulate(
            factor_table, headers=string_header, tablefmt=tablefmt, floatfmt=".4f"
        )


# variable elimination module. Adapted from
# http://pgmpy.org/_modules/pgmpy/inference/ExactInference.html#VariableElimination
class VE:
    def __init__(self, factors):
        """
        Inputs:
            factors: list of Factor
        Outputs: sets the following instance variables
        """
        # get working factors i.e.m the list of potentials that each variable
        # participates in
        self.factors = defaultdict(list)
        for factor in factors:
            for var in factor.variables:
                self.factors[var].append(factor)
        return

    def _get_working_factors(self):
        """
        Make copy of working factors.
        """
        working_factors = {
            node: [factor for factor in self.factors[node]] for node in self.factors
        }
        return working_factors

    def _variable_elimination(
            self,
            variables,
            elimination_order,
            joint,
            show_progress,
    ):
        """
        Implementation of a generalized variable elimination.

        Parameters
        ----------
        variables: list, array-like
            variables that are not to be eliminated.

        elimination_order: list (array-like)
            variables in order of being eliminated 
        """

        operation = 'marginalize'

        # Step 1: Deal with the input arguments.
        if isinstance(variables, str):
            raise TypeError("variables must be a list of strings")

        # Step 2: Prepare data structures to run the algorithm.
        eliminated_variables = set()
        # Get working factors
        working_factors = self._get_working_factors()

        # Step 3: Run variable elimination
        if show_progress:
            pbar = tqdm(elimination_order)
        else:
            pbar = elimination_order

        max_factor_size = 0
        for var in pbar:
            if show_progress:
                pbar.set_description("Eliminating: {var}".format(var=var))
            # Removing all the factors containing the variables which are
            # eliminated (as all the factors should be considered only once)
            factors = [
                factor
                for factor in working_factors[var]
                if not set(factor.variables).intersection(eliminated_variables)
            ]
            phi = factors[0].many_products(factors[1:])
            phi = getattr(phi, operation)([var], inplace=False)
            max_factor_size = max(max_factor_size, len(phi.scope()))
            """
            if show_progress:
                print("Eliminated variable was %s" %var)
                print("\tResulting factor is")
                print(phi)
            """
            del working_factors[var]
            for variable in phi.variables:
                working_factors[variable].append(phi)
            eliminated_variables.add(var)

        if show_progress:
            print("Maximum clique formed %d" % max_factor_size)

        # Step 4: Prepare variables to be returned.
        final_distribution = []
        for node in working_factors:
            factors = working_factors[node]
            for factor in factors:
                if not set(factor.variables).intersection(eliminated_variables):
                    final_distribution.append(factor)

        if joint:
            return final_distribution[0].many_products(final_distribution[1:])
        else:
            query_var_factor = {}
            phi = final_distribution[0].many_products(final_distribution[1:])
            for query_var in variables:
                query_var_factor[query_var] = phi.marginalize(
                    list(set(variables) - set([query_var])), inplace=False
                )
            return query_var_factor

    def query(
            self,
            variables,
            elimination_order,
            joint,
            show_progress,
    ):
        """
        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability

        elimination_order: list
            order of variable eliminations.

        joint: boolean (default: True)
            If True, returns a Joint Distribution over `variables`.
            If False, returns a dict of distributions over each of the `variables`.
        """
        return self._variable_elimination(
            variables=variables,
            elimination_order=elimination_order,
            joint=joint,
            show_progress=show_progress,
        )

    def get_lognorm(self, elimination_order, show_progress):
        """
        Compute log normalizer for the joint distribution.
        Inputs:
            elimination_order:
            show_progress: boolean, whether to print progress of VE
        """
        last = elimination_order[-1]
        phi = self.query(variables=[last], elimination_order=elimination_order[:-1], joint=True,
                         show_progress=show_progress)
        logZ = logsumexp(phi.log_values)
        return logZ


# elimination order module. Adapted from
# http://pgmpy.org/_modules/pgmpy/inference/EliminationOrder.html#BaseEliminationOrder
class BaseEliminationOrder:
    """
    Base class for finding elimination orders.
    """

    def __init__(self, G):
        """
        Init method for the base class of Elimination Orders.
        Parameters
        ----------
        G: NetworkX graph 
        """
        self.G = G.copy()
        return

    def cost(self, node):
        """
        The cost function to compute the cost of elimination of each node.
        This method is just a dummy and returns 0 for all the nodes. Actual cost functions
        are implemented in the classes inheriting BaseEliminationOrder.
        Parameters
        ----------
        node: string, any hashable python object.
            The node whose cost is to be computed.
        """
        return 0

    def get_elimination_order(self, nodes=None, show_progress=True):
        """
        Returns the optimal elimination order based on the cost function.
        The node having the least cost is removed first.
        Parameters
        ----------
        nodes: list, tuple, set (array-like)
            The variables which are to be eliminated.
        """
        nodes = self.G.nodes()

        ordering = []
        if show_progress:
            pbar = tqdm(total=len(nodes))
            pbar.set_description("Finding Elimination Order: ")

        while len(self.G.nodes()) > 0:
            # find minimum score node
            scores = {node: self.cost(node) for node in self.G.nodes()}
            min_score_node = min(scores, key=scores.get)
            # add found node to elimination order
            ordering.append(min_score_node)
            # add edges to node's neighbors
            edge_list = self.fill_in_edges(min_score_node, show_progress)
            self.G.add_edges_from(edge_list)
            # remove node from graph
            self.G.remove_node(min_score_node)
            if show_progress:
                pbar.update(1)
        return ordering

    def fill_in_edges(self, node, show_progress=False):
        """
        Return edges needed to be added to the graph if a node is removed.
        Parameters
        ----------
        node: string (any hashable python object)
            Node to be removed from the graph.
        show_progress: boolean, print clique size formed after removal of 
            vertex
        """
        neighbors = list(self.G.neighbors(node))
        degree = len(neighbors)
        edge_list = []
        if (show_progress):
            print("After removing %s, a clique of size %d forms" % (node, degree))
        if (degree > 1):
            for i in range(degree):
                for j in range(degree - 1):
                    if not self.G.has_edge(neighbors[i], neighbors[j]):
                        edge_list.append((neighbors[i], neighbors[j]))

        return edge_list


class MinFill(BaseEliminationOrder):
    def cost(self, node):
        """
        The cost of a eliminating a node is the number of edges that need to be added
        (fill in edges) to the graph due to its elimination
        """
        return len(self.fill_in_edges(node))


# Potts() -----------------------------------------------------------------------
class Potts(AbstractModel):
    def __init__(self, data, config_dict):
        """
        Set up MRF, do some processing to get rough estimate of class means,
        use a heuristic to find a good variable elimination order, report 
        the maximum clique size formed during elimination.

        Inputs:
            data: dictionary, keys being counts y and adjacency matrix W
            config_dict: dictionary with following keys
                beta: scalar, connectivity of lattice MRF
                heuristic: str, type of heuristic to determine elimination order, MinDegree or MinFill
                display: boolean, whether to print information about MRF
        """
        self._y = data['y']
        self._N = len(self._y)
        self._W = np.asarray(data['W'], dtype=int)
        self._beta = config_dict["beta"]
        self._lowmean = np.quantile(self._y, 0.25)
        self._highmean = np.quantile(self._y, 0.5)
        display = config_dict["display"]

        if display:
            print("There are %d sites" % len(self._y))
            print("Dimensions of adjacency matrix (%d, %d)" % (self._W.shape[0], self._W.shape[1]))
            list_of_counts = ["(site %d, y %d)" % (i, self._y[i]) for i in range(self._N)]
            print("Counts")
            print(list_of_counts)
            print("Adjacency matrix")
            print(self._W)
            print("Initial guess of class means %.2f and %.2f" % (self._lowmean, self._highmean))

        # use heuristic to find a good elimination order
        heuristic = config_dict["heuristic"]
        if heuristic == "MinDegree":
            degrees = np.sum(self._W, axis=1)
            nodes = range(self._N)
            temp = list(zip(nodes, degrees))
            temp = sorted(temp, key=lambda tup: tup[1])
            elimination_order = ["x%d" % d[0] for d in temp]
        else:
            G = nx.from_numpy_matrix(self._W)
            mapping = {i: "x%d" % i for i in range(len(G.nodes()))}
            G = nx.relabel_nodes(G, mapping)
            nodes = G.nodes()
            if (heuristic == "MinFill"):
                elimination_order = MinFill(G).get_elimination_order(nodes, display)
            elif (heuristic == "MinNeighbors"):
                elimination_order = MinNeighbors(G).get_elimination_order(nodes, display)
        self._elimination_order = elimination_order

        if display:
            print("Variable elimination will use the following order")
            print(elimination_order)

        # check size of biggest clique encountered during elimination
        factors = []
        # unary potentials (in case some nodes are disconnected from other nodes)
        for i in range(self._N):
            node0 = "x%d" % (i)
            factor = Factor([node0], cardinality=[2], log_values=[0, 0])
            factors.append(factor)

        # pairwise potentials
        for i in range(self._N):
            for j in range(self._N):
                if self._W[i, j] == 1:
                    node0 = "x%d" % (i)
                    node1 = "x%d" % (j)
                    factor = Factor([node0, node1], cardinality=[2, 2], log_values=[self._beta, 0, 0, self._beta])
                    factors.append(factor)

        inference = VE(factors)
        t0 = time.time()
        self._logZ = inference.get_lognorm(elimination_order, display)
        t1 = time.time()

        if display:
            print("Time of one normalization constant computation %.2f" % (t1 - t0))
            print("Finished initialization\n")
        return

    def _get_elimination_order(self):
        """
        Return copy of elimination order.
        """
        return self._elimination_order.copy()

    def get_neighbors(self, i):
        list_of_neighbors = ["(site %d, y %d)" % (j, self._y[j]) for j in range(self._N) if self._W[i, j] == 1]
        return list_of_neighbors

    def fit(self, r_init, display, max_iter=100, var_converge=1e-8):
        """
        Estimate class means using full data and compute relevant sensitivities for 
        IJ approximation. Report class means and relevant runtimes.
        """
        # t0 = time.time()
        if r_init is None:
            r_init = np.array([self._lowmean, self._highmean])
        params_ones = self.EM(r_init, display, max_iter, var_converge)
        # t1 = time.time()
        # print("Finished fitting in %.2f seconds" %(t1-t0))
        return params_ones

    # -----------------------------------------------------------------------------------------
    # EM code
    def EM(self, r_init, display, max_iter, var_converge):
        """
        EM to maximize log(y; r0, r1) w.r.t. r0 and r1
        Input:
            r_init: list, class_means
        """
        # initialize using lowmean and highmean
        params = r_init
        LLlog = []
        iteration = 0
        prevLL = self.LL(params)
        LLlog.append(prevLL)
        while True:
            iteration += 1
            q = self.Estep(params)
            r = self.Mstep(q)
            params = r
            LL = self.LL(params)
            LLlog.append(LL)
            if (iteration > max_iter or abs((LL - prevLL) / prevLL) < var_converge):
                break
            prevLL = LL
        if display:
            # plot log LL of data as function of EM iteration
            plt.figure()
            plt.plot(range(len(LLlog)), LLlog, marker='o')
            plt.xlabel("Iteration", fontsize=15)
            plt.ylabel("Log-likelihood", fontsize=15)
            plt.tick_params(axis='x', labelsize=15)
            plt.tick_params(axis='y', labelsize=15)
            plt.show()
            # report final class means
            print("Inital means of EM r0 = %.2f, r1 = %.2f" % (r_init[0], r_init[1]))
            print("\tFinal means of EM r0 = %.2f, r1 = %.2f" % (r[0], r[1]))
        return r

    def Estep(self, params, printfactors=False):
        """
        Return the marginal distributions p(x_i|y;params). Because binary variable,
        only report p(x_i=0|y;params).
        """
        r0 = params[0]
        r1 = params[1]

        # create list of factors representing p(x|y;params)
        factors = []

        # unary potentials in the prior (in case some nodes are disconnected from other nodes)
        for i in range(self._N):
            node0 = "x%d" % (i)
            factor = Factor([node0], cardinality=[2], log_values=[0, 0])
            factors.append(factor)

        ## pairwise potentials in the prior 
        for i in range(self._N):
            for j in range(self._N):
                if self._W[i, j] == 1:
                    node0 = "x%d" % (i)
                    node1 = "x%d" % (j)
                    prior_factor = Factor([node0, node1], cardinality=[2, 2], log_values=[self._beta, 0, 0, self._beta])
                    factors.append(prior_factor)

        ## add observations to make the joint model
        for i in range(self._N):
            node = "x%d" % (i)
            low_class = -r0 + self._y[i] * np.log(r0) - gammaln(self._y[i] + 1)
            high_class = -r1 + self._y[i] * np.log(r1) - gammaln(self._y[i] + 1)
            observation_factor = Factor([node], cardinality=[2], log_values=[low_class, high_class])
            factors.append(observation_factor)

        inference = VE(factors)

        # marginals p(x_i|y;params)
        q = np.zeros(self._N)  # q(x_i^0)
        for i in range(self._N):
            node = "x%d" % i
            elimination_order = self._get_elimination_order()
            elimination_order.remove(node)
            marginal_factor = inference.query(variables=[node], elimination_order=elimination_order, joint=True,
                                              show_progress=False)
            log_norm = logsumexp(marginal_factor.log_values)
            # properly normalize the marginal factors
            marginal_factor.log_values = marginal_factor.log_values - log_norm
            if (printfactors):
                print(marginal_factor)
            q[i] = np.exp(marginal_factor.log_values[0])
        return q

    def Mstep(self, low):
        """
        Update class means.
        Inputs: 
            low: list, marginal distributions q(x_i = 0|y;params) 
        """
        high = 1 - low
        r0 = np.sum(low * self._y) / (np.sum(low) + 1e-100)
        r1 = np.sum(high * self._y) / (np.sum(high) + 1e-100)
        r = np.array([r0, r1])
        return r

    # regular log likelihood 
    def LL(self, params):
        """
        Compute log p(y;params) = log sum_x p(y,x;params) 
        where 
            - p(y|x) is product of independent Poisson with two classes of means r0, r1 
            - p(x) is Potts model with uniform connection strength beta: 
            p(x) propto exp(beta sum_{i,j} W_{ij} {x_i = x_j})
        Inputs:
            params: list - r0, r1
        """
        r0 = params[0]
        r1 = params[1]

        # create list of factors representing p(x|y;params)
        factors = []

        # unary potentials in the prior (in case some nodes are disconnected from other nodes)
        for i in range(self._N):
            node0 = "x%d" % (i)
            factor = Factor([node0], cardinality=[2], log_values=[0, 0])
            factors.append(factor)

        ## iterate through all edges to create the prior model
        for i in range(self._N):
            for j in range(self._N):
                if self._W[i, j] == 1:
                    node0 = "x%d" % (i)
                    node1 = "x%d" % (j)
                    prior_factor = Factor([node0, node1], cardinality=[2, 2], log_values=[self._beta, 0, 0, self._beta])
                    factors.append(prior_factor)

        ## add observations to make the joint model
        for i in range(self._N):
            node = "x%d" % (i)
            low_class = -r0 + self._y[i] * np.log(r0) - gammaln(self._y[i] + 1)
            high_class = -r1 + self._y[i] * np.log(r1) - gammaln(self._y[i] + 1)
            observation_factor = Factor([node], cardinality=[2], log_values=[low_class, high_class])
            factors.append(observation_factor)

        inference = VE(factors)
        elimination_order = self._get_elimination_order()
        log_numerator = inference.get_lognorm(elimination_order, show_progress=False)
        return log_numerator - self._logZ

    # -------------------------------------------------------------------------------------------
    # IJ code
    def weighted_loss(self, params, weights):
        """
        Compute -log p(y;weights,params) = -log sum_x p(y,x;weights,params) 
        where 
            - p(y|x; weights) is product of independent Poisson with two classes of means r0, r1 
            over present observations i.e. weights[i] = 1.
            - p(x) is Potts model with uniform connection strength beta: 
            p(x) propto exp(beta sum_{i,j} weights[i] * weights[j] W_{ij} {x_i = x_j})

        Inputs:
            params: list - r0, r1
            weights: list, whether sites are present in model
        """
        r0 = params[0]
        r1 = params[1]
        elimination_order = self._get_elimination_order()

        denom_factors = []
        # iterate through all edges to create the prior model
        for i in range(self._N):
            for j in range(self._N):
                if self._W[i, j] == 1:
                    node0 = "x%d" % (i)
                    node1 = "x%d" % (j)
                    inclusion_weight = weights[i] * weights[j]
                    log_values = inclusion_weight * np.array([self._beta, 0, 0, self._beta])
                    prior_factor = Factor([node0, node1], cardinality=[2, 2], log_values=log_values)
                    denom_factors.append(prior_factor)

        prior_inference = VE(denom_factors)
        log_denominator = prior_inference.get_lognorm(elimination_order, show_progress=False)

        # add the observations to make the joint model
        numer_factors = denom_factors.copy()
        for i in range(self._N):
            node = "x%d" % (i)
            low_class = -r0 + self._y[i] * np.log(r0) - gammaln(self._y[i] + 1)
            high_class = -r1 + self._y[i] * np.log(r1) - gammaln(self._y[i] + 1)
            log_values = weights[i] * np.array([low_class, high_class])
            observation_factor = Factor([node], cardinality=[2], log_values=log_values)
            numer_factors.append(observation_factor)

        joint_inference = VE(numer_factors)
        log_numerator = joint_inference.get_lognorm(elimination_order, show_progress=False)
        weighted_LL = log_numerator - log_denominator
        final_weighted_loss = -weighted_LL
        return final_weighted_loss

    # -------------------------------------------------------------------------------------------------
    # predictive code
    def loo_predictive(self, missing_site, params, display=False):
        """
        Compute log p(y_i|y_{-i};params).
        Inputs:
            missing_site: scalar, location of missing site
            params: list, class means
            display: boolean, whether to plot factors 
        Outputs:
            log likelihood and how much time it took to compute
        """
        t0 = time.time()
        # compute log p(x|y_{-i};params)
        r0 = params[0]
        r1 = params[1]

        numer_factors = []

        # unary potentials in the prior (in case some nodes are disconnected from other nodes)
        for i in range(self._N):
            node0 = "x%d" % (i)
            factor = Factor([node0], cardinality=[2], log_values=[0, 0])
            numer_factors.append(factor)

        ## iterate through all edges to create the prior model
        for i in range(self._N):
            for j in range(self._N):
                if self._W[i, j] == 1:
                    node0 = "x%d" % (i)
                    node1 = "x%d" % (j)
                    prior_factor = Factor([node0, node1], cardinality=[2, 2], log_values=[self._beta, 0, 0, self._beta])
                    numer_factors.append(prior_factor)

        # add non-missing observations to make the joint model
        for i in range(self._N):
            if i != missing_site:
                node = "x%d" % (i)
                low_class = -r0 + self._y[i] * np.log(r0) - gammaln(self._y[i] + 1)
                high_class = -r1 + self._y[i] * np.log(r1) - gammaln(self._y[i] + 1)
                observation_factor = Factor([node], cardinality=[2], log_values=[low_class, high_class])
                numer_factors.append(observation_factor)

        inference = VE(numer_factors)
        # compute log p(x_i|y_{-i};params) by marginalizing x_{-i} in log p(x|y_{-i};params)
        node = "x%d" % missing_site
        elimination_order = self._get_elimination_order()
        elimination_order.remove(node)
        marginal_factor = inference.query(variables=[node], elimination_order=elimination_order, joint=True,
                                          show_progress=False)
        log_norm = logsumexp(marginal_factor.log_values)
        marginal_factor.log_values = marginal_factor.log_values - log_norm

        if display:
            print("Site %d missing. p(x_i|y_{-i}) is" % missing_site)
            print(marginal_factor)

        # compute log p(y_i|y_{-i};params)
        low_class = -r0 + self._y[missing_site] * np.log(r0) - gammaln(self._y[missing_site] + 1)
        high_class = -r1 + self._y[missing_site] * np.log(r1) - gammaln(self._y[missing_site] + 1)
        loglowterm = marginal_factor.log_values[0] + low_class
        loghighterm = marginal_factor.log_values[1] + high_class
        ans = logsumexp([loglowterm, loghighterm])
        t1 = time.time()
        return ans, t1 - t0
