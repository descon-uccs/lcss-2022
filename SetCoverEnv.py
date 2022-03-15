from __future__ import annotations
from abc import abstractmethod, ABC
import random
from math import exp
from EnvBase import Agent
from SetCover import Resource


class Formatter:
    none_str = '(none)'

    @staticmethod
    def indent(string: str, indent_size: int = 2):
        lines = string.splitlines(True)
        i = " " * indent_size
        lines = [i + line for line in lines]
        return "".join(lines)


class SetCoverAction:
    def __init__(self, parent: SetCoverAgent):
        self._resources = list()
        self._agent = parent

    @property
    def resources(self):
        for resource in self._resources:
            yield resource

    @property
    def utility(self):
        return self._agent.utility_function.value_wrt(self)

    def __str__(self):
        return self.SetCoverActionFormatter.format(self)

    @staticmethod
    def build(owner: SetCoverAgent, parent_builder: SetCoverAgent.Builder):
        return SetCoverAction.Builder(owner, parent_builder)

    class Builder:
        def __init__(self, owner: SetCoverAgent, parent_builder: SetCoverAgent.Builder):
            self.__parent = parent_builder
            self.__action = SetCoverAction(owner)

        def add_resource(self, resource):
            if resource not in self.__action._resources:
                self.__action._resources.append(resource)
            else:
                print(f"Resource {resource.name} already in action.")

        def build(self):
            self.__action._resources.sort(key=lambda resource: resource.name)
            if self.__parent:
                self.__parent.add_action(self.__action)
            return self.__action

    class SetCoverActionFormatter(Formatter):
        @classmethod
        def format(cls, action: SetCoverAction):
            elements = list()
            for resource in action._resources:
                elements.append(f"{resource.name}")
            return ', '.join(elements)


class SetCoverAgent(Agent):
    def __init__(self, name: str, utility_function_factory: UtilityFunctionFactory, **kwargs):
        super().__init__(name)
        self._utility_function = utility_function_factory.create_function_instance(self)
        self._action_set = list()
        self._selected_action = None
        self._history = dict()
        self._q_table = dict()

    def do_selection(self, selector):
        old_action = self.selected_action
        self._selected_action = selector.select_from(self._action_set)
        if self.selected_action is not old_action:
            if old_action:
                for resource in old_action.resources:
                    resource.remove(self)
            if self._selected_action:
                for resource in self.selected_action.resources:
                    resource.add(self)

    def do_q_selection(self, random_threshold):
        old_action = self.selected_action
        if random.random() > random_threshold:
            # select the action with the highest q-value
            current = 0.0
            best_responses = list()
            for action in self._action_set:
                q_value = self._q_table.get(action, 0.0)
                if q_value > current:
                    current = q_value
                    best_responses.clear()
                    best_responses.append(action)
                elif q_value == current:
                    best_responses.append(action)
            self._selected_action = random.choice(best_responses)
        else:
            # select an action uniformly at random
            self._selected_action = random.choice(self._action_set)
        if self.selected_action is not old_action:
            if old_action:
                for resource in old_action.resources:
                    resource.remove(self)
            if self._selected_action:
                for resource in self.selected_action.resources:
                    resource.add(self)

    def incorporate_feedback(self, feedback):
        if feedback:
            # print("test")
            self._history[self._selected_action] = self._history.get(self._selected_action, 0) + 1

    def update_q_table(self):
        # create a set of actions with the least numbers of negative feedbacks
        min_feedback = 10000
        best_responses = list()
        for action in self._action_set:
            feedbacks = self._history.get(action, 0)
            if feedbacks < min_feedback:
                best_responses.clear()
                best_responses.append(action)
                min_feedback = feedbacks
            elif feedbacks == min_feedback:
                best_responses.append(action)
        q_best = random.choice(best_responses)
        for action in self._action_set:
            if action is q_best:
                self._q_table[action] = 1.0
            else:
                self._q_table[action] = 0.0


    @property
    def utility_function(self) -> UtilityFunction:
        return self._utility_function

    @property
    def selected_action(self):
        return self._selected_action

    @property
    def actions(self):
        for action in self._action_set:
            yield action

    def __str__(self):
        return self.SetCoverAgentFormatter.format(self)

    class Builder:
        def __init__(self,
                     agent_name: str,
                     utility_function_factory: UtilityFunctionFactory,
                     parent_builder: SetCoverGame.Builder = None):
            self.__parent = parent_builder
            self.__agent = SetCoverAgent(agent_name, utility_function_factory)

        def build(self):
            if self.__parent:
                self.__parent.add_agent(self.__agent)
            return self.__agent

        def create_action(self):
            return SetCoverAction.Builder(self.__agent, self)

        def add_action(self, action: SetCoverAction):
            self.__agent._action_set.append(action)

    class SetCoverAgentFormatter(Formatter):
        @classmethod
        def format(cls, agent: SetCoverAgent):
            lines = list()
            lines.append(str(agent))
            if agent._action_set:
                lines.append(cls.indent('Actions:'))
                for action in agent._action_set:
                    lines.append(cls.indent(str(action)))
            return '\n'.join(lines)


class SetCoverResource(Resource):
    def __init__(self, name: str, value: float):
        super().__init__(name, value)
        self._coverage = set()

    def add(self, agent: Agent):
        self._coverage.add(agent)

    def remove(self, agent: Agent):
        self._coverage.remove(agent)

    @property
    def is_covered(self):
        if self._coverage:
            return True
        return False

    @property
    def coverage(self):
        for resource in self._coverage:
            yield resource

    def __str__(self):
        return self.SetCoverResourceFormatter.format(self, super().__str__())

    class SetCoverResourceFormatter(Formatter):
        @classmethod
        def format(cls, res: SetCoverResource, parent_str: str) -> str:
            lines = parent_str.splitlines(False)
            if res._coverage:
                lines.append(cls.indent('Covered by:'))
                agents = list()
                for agent in res._coverage:
                    agents.append(agent.name)
                agents = cls.indent(', '.join(agents))
                lines.append(agents)
            return '\n'.join(lines)


class ActionSelector(ABC):
    @abstractmethod
    def select_from(self, action_set: list) -> SetCoverAction:
        pass


class BestResponseSet(ActionSelector):
    def __init__(self, current_action):
        self._selected_action = current_action

    def select_from(self, action_set: list) -> SetCoverAction:
        pass


class RandomSelector(ActionSelector):
    def select_from(self, action_set):
        return random.choice(action_set)


class LogLinear(ActionSelector):
    def __init__(self, temperature):
        self._temperature = temperature

    def select_from(self, action_set):
        probabilities = dict()
        for action in action_set:
            probabilities.update({action: exp(action.utility / self._temperature)})
        denominator = sum(probabilities.values())
        rand = random.random() * denominator
        accumulator = 0.0
        count: int = 0
        for action_weight in sorted(probabilities.items(), key=lambda kv: kv[1]):
            accumulator += action_weight[1]
            count += 1
            if accumulator > rand or count == len(probabilities):
                return action_weight[0]


class BestResponse(ActionSelector):
    def select_from(self, action_set) -> SetCoverAction:
        pass


class SetCoverGame:
    def __init__(self):
        self._agents = dict()             # Set of agents
        self._resources = dict()          # Set of resources
        self._information_edges = dict()  # Set of directed information edges
        self._simulator = None

    @staticmethod
    def build(utility_function_factory):
        return SetCoverGame.Builder(utility_function_factory)

    def agent(self, name):
        return self._agents.get(name)

    def resource(self, name):
        return self._resources.get(name)

    @property
    def simulator(self):
        if not self._simulator:
            self._simulator = SetCoverGame.Simulator(self)
        return self._simulator

    @property
    def resources(self):
        for resource in self._resources:
            yield resource

    def __str__(self):
        return self.SetCoverGameFormatter.format(self)

    class Builder:
        def __init__(self, utility_function_factory: type(UtilityFunctionFactory)):
            self.__game = SetCoverGame()
            self._utility_function_factory = utility_function_factory.get_instance(self)

        def agent(self, name):
            return self.__game.agent(name)

        def resource(self, name):
            return self.__game.resource(name)

        def add_resource(self, resource):
            self.__game._resources.update({resource.name: resource})

        def create_resource(self, name: str, value: float) -> Resource:
            new_resource = SetCoverResource(name, value)
            self.add_resource(new_resource)
            return new_resource

        def add_agent(self, agent: SetCoverAgent):
            self.__game._agents.update({agent.name: agent})
            self.__game._information_edges.update({agent: set()})

        def create_agent(self, name):
            return SetCoverAgent.Builder(name, self._utility_function_factory, self)

        def add_information_edge(self, source: Agent, target: Agent):
            self.__game._information_edges.get(target).add(source)

        @property
        def information_edges(self):
            return self.__game._information_edges

        def build(self):
            return self.__game

    class SetCoverGameFormatter(Formatter):
        @classmethod
        def format(cls, game: SetCoverGame) -> str:
            lines = list()
            lines.append("Game:")

            # Resources
            resources_lines = list()
            if game._resources:
                for resource in game._resources.values():
                    resources_lines.append(cls.indent(str(resource)))
            else:
                resources_lines.append(cls.indent(cls.none_str))
            resources_lines.insert(0, "Resources:")
            lines.append(cls.indent("\n".join(resources_lines)))

            # Agents
            agents_lines = list()
            if game._agents:
                for agent in game._agents.values():
                    agents_lines.append(cls.indent(str(agent)))
            else:
                agents_lines.append(cls.indent(cls.none_str))
            agents_lines.insert(0, 'Agents:')
            lines.append(cls.indent('\n'.join(agents_lines)))

            # Information Edges
            information_edges_lines = list()
            if game._information_edges:
                for item in game._information_edges.items():
                    information_edges_lines.append(f'Target Node: {item[0].name}')
                    sources = [res.name for res in item[1]]
                    if sources:
                        sources = ', '.join(sources)
                    else:
                        sources = cls.none_str
                    information_edges_lines.append(cls.indent(f'Source Nodes: {sources}'))
            else:
                information_edges_lines.append(cls.none_str)
            lines.append(cls.indent('Information Edges:\n' + cls.indent('\n'.join(information_edges_lines))))
            return "\n".join(lines)

    class Simulator:
        def __init__(self, game: SetCoverGame):
            self._game = game
            self._selector = None
            self._t: int = 0
            self._decay = None

        def initialize(self, selector: ActionSelector, seed=0, decay=.002):
            self._selector = selector
            self._t = 0
            random.seed(seed)
            self._decay = decay
            random_selector = RandomSelector()
            agent: SetCoverAgent
            for agent in self._game._agents.values():
                agent.do_selection(random_selector)

        def update_q_tables(self):
            for agent in self._game._agents.values():
                agent.update_q_table()

        def next(self):
            self._t += 1
            agent: SetCoverAgent
            agent = random.choice(list(self._game._agents.values()))
            if self._selector is None:
                agent.do_q_selection(exp(-1.0 * self._decay * self._t))
            else:
                agent.do_selection(self._selector)
            return agent

        def objective_function_value(self):
            resource: SetCoverResource
            value: float = 0
            for resource in self._game._resources.values():
                if resource.is_covered:
                    value += resource.value
            return value

        @property
        def t(self):
            return self._t


class UtilityFunction(ABC):
    @abstractmethod
    def value_wrt(self, action: SetCoverAction) -> float:
        pass


class UtilityFunctionFactory(ABC):
    def create_function_instance(self, agent: SetCoverAgent) -> UtilityFunction:
        pass

    @staticmethod
    def get_instance(gb: SetCoverGame.Builder):
        pass


class MarginalValue(UtilityFunction):
    _information_edges = None

    def __init__(self, agent: SetCoverAgent, information_edges: dict):
        self._information_edges = information_edges
        self._agent = agent

    def value_wrt(self, action: SetCoverAction):
        resource: SetCoverResource
        total_utility = 0.0
        source_nodes = self._information_edges.get(self._agent)
        for resource in action.resources:
            if resource.value > 0:
                resource_utility = resource.value
                for covering_agent in resource.coverage:
                    if covering_agent in source_nodes:
                        resource_utility = 0.0
                        break
                total_utility += resource_utility
        return total_utility

    class Factory(UtilityFunctionFactory):
        def __init__(self, information_edges):
            self._information_edges = information_edges

        @staticmethod
        def get_instance(gb: SetCoverGame.Builder):
            return MarginalValue.Factory(gb.information_edges)

        def create_function_instance(self, agent: SetCoverAgent):
            return MarginalValue(agent, self._information_edges)


class EqualShare(UtilityFunction):
    _information_edges = None

    def __init__(self, agent: SetCoverAgent, information_edges: dict):
        self._information_edges = information_edges
        self._agent = agent

    def value_wrt(self, action: SetCoverAction):
        resource: SetCoverResource
        total_utility = 0.0
        source_nodes = self._information_edges.get(self._agent)
        for resource in action.resources:
            if resource.value > 0.0:
                num_covering = 1
                for covering_agent in resource.coverage:
                    if covering_agent in source_nodes:
                        num_covering += 1
                        break
                total_utility += resource.value / num_covering
        return total_utility

    class Factory(UtilityFunctionFactory):
        def __init__(self, information_edges):
            self._information_edges = information_edges

        @staticmethod
        def get_instance(gb: SetCoverGame.Builder):
            return EqualShare.Factory(gb.information_edges)

        def create_function_instance(self, agent: SetCoverAgent):
            return EqualShare(agent, self._information_edges)
