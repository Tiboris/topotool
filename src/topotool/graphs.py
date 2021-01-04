#!/usr/bin/env python3

from collections import OrderedDict
from copy import Error


class Vertex:
    def __init__(self, vertexid):
        self.vertexid = vertexid
        self.name = vertexid
        self.connections = set()

    def has_connection_with(self, vertexid):
        return vertexid in self.connections

    def connect(self, vertexid):
        self.connections.add(vertexid)

    @property
    def conn_cnt(self):
        return len(self.connections)

    @property
    def neighbours(self):
        return self.connections

    def disconnect(self, vertexid):
        try:
            self.connections.remove(vertexid)
        except KeyError:
            pass

    def __str__(self):
        res = "---VERTEX---\n"
        res += f"{self.vertexid}\n"
        if self.conn_cnt:
            tmpl = list(self.connections)
            if self.conn_cnt > 1:
                for idx in range(self.conn_cnt - 1):
                    res += f"├─ {tmpl[idx]}\n"
            res += f"└─ {tmpl[-1]}\n"
        res += "------------"
        return res


class Edge:
    def __init__(self, vertex_a, vertex_b, weight=0, oriented=False):
        self.data = tuple([vertex_a.name, vertex_b.name])
        if not oriented:
            self.rdata = tuple([vertex_b.name, vertex_a.name])
        self.weight = weight

    def __str__(self):
        conn = "---" if not self.weight else "-" + self.weight + "-"
        return self.vertex_a.name + conn + self.vertex_a.name


class Graph:
    def __init__(
        self, data, Instance=Vertex,
        weight_delim="", data_format="ipa",
    ):
        self.Instance = Instance
        self.oriented = False
        self.vertices = OrderedDict()
        self._edges = {}
        if data_format == "ipa":
            delim = "|"
            self._parse_ipa_segments(data, delimiter=delim)
        elif data_format == "edges":
            delim = "-"
            self._parse_vertices(data, delimiter=",")
            self._parse_edge_connections(data, delim, weight_delim)
        else:
            raise NotImplementedError(
                "Unknown format of the data in file selected."
            )

        self.delimiter = delim

    def __str__(self):
        res = "+++GRAPH++++\n"
        for key in self.vertices:
            res += f"{self.vertices[key]}\n"
        res += "+++++++++++*\n"
        return res

    @property
    def edge_list(self):
        return [self._edges[key].data for key in self._edges]

    def _parse_vertices(self, data, delimiter=","):
        vertices = list()

        if delimiter in data[0]:
            for vertex in data[0].split(delimiter):
                vertices.append(vertex)

            del data[0]

            for key in vertices:
                self.vertices[key] = self.Instance(key)

    def _parse_ipa_segments(self, data, delimiter="|"):
        """
        Parse edges from `ipa topologysegment_find domain` command output
        """
        edges = {}
        left_node = ""
        right_node = ""

        if "---" in data[0]:
            expected_seg_cnt = int(data[1].split(" segments matched")[0])
            # delete output header in format:
            # ------------------
            # X segments matched
            # ------------------
            for _ in range(3):
                del data[0]
        else:
            raise Error(
                "Unexpected file content, FreeIPA input "
                " pareser expects '---' at file start"
            )

        segments_cnt = 0

        for line in data:
            if "---" in line:
                break

            # get rid of trailing whitespaces
            line = line.strip()
            if not line or "Segment name:" in line:
                continue

            key, value = line.split(": ")
            # print(key, value)
            if key == "Left node":
                left_node = value
                # print("found:", left_node)

            if key == "Right node":
                right_node = value
                # print("found:", right_node)

            if key == "Connectivity":
                if value == "both":
                    conn = left_node + delimiter + right_node
                    if left_node not in self.vertices:
                        self.vertices[left_node] = self.Instance(left_node)
                    if right_node not in self.vertices:
                        self.vertices[right_node] = self.Instance(right_node)

                    edges[conn] = Edge(
                        self.vertices[left_node], self.vertices[right_node]
                    )
                    self.vertices[left_node].connect(right_node)
                    self.vertices[right_node].connect(left_node)
                    # print("connecting", left_node, right_node)
                    left_node = ""
                    right_node = ""
                    segments_cnt += 1
                else:
                    raise NotImplementedError(
                        f"Connectivity '{value}' is not supported"
                    )

        if not segments_cnt:
            raise Error("Did not find any segments")
        if segments_cnt != expected_seg_cnt:
            raise Error(
                f"Expected {expected_seg_cnt} segments "
                f"but {segments_cnt} parsed"
            )

        self._edges = edges

    def _parse_edge_connections(self, data, delimiter, weight_delim):
        edges = {}
        for connection in data:
            weight = 0
            if weight_delim:
                connection, weight = connection.split(weight_delim)

            a, b = connection.split(delimiter)

            if a in self.vertices and b in self.vertices:
                self.oriented = ">" in delimiter

                edges[connection] = Edge(
                    self.vertices[a], self.vertices[b],
                    weight, oriented=self.oriented,
                )

                self.vertices[a].connect(b)
                if not self.oriented:
                    self.vertices[b].connect(a)

        self._edges = edges

    def empty(self):
        self._edges = {}
        for key in self.vertices:
            self.vertices[key].connections = set()

    def disconnect(self, a, b):
        self.vertices[a].disconnect(b)
        self.vertices[b].disconnect(a)
        try:
            del self._edges[a + self.delimiter + b]
        except KeyError:
            pass

    def remove_edge(self, edge):
        a, b = edge.split(self.delimiter)
        self.disconnect(a, b)

    def neighbours(self, objvertex):
        res = set()
        for obj in objvertex:
            for neighbour in obj.connections:
                res.add(self.vertices[neighbour])

        return res

    def remove_vertex(self, vertex):
        try:
            del self.vertices[vertex.vid]
        except KeyError:
            pass

        to_del = [edge for edge in self._edges if vertex.id in edge]

        for edge in to_del:
            self.remove_edge(edge)

    def vertex_cnt(self):
        return len(self.vertices)

    def edge_cost(self, a, b):  # rework edge to be dict
        for edge in self._edges:
            if a in edge and b in edge:
                print(edge)
                return self._edges[edge].weight

        return -1  # returns -1 if given vertex names are not in graph
