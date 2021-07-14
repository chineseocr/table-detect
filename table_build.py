#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 17:31:37 2021

@author: lywen
"""


class tableBuid:
    ##表格重建
    def __init__(self, ceilbox, interval=10):
        """
        ceilboxes:[[x0,y0,x1,y1,x2,y2,x3,y3,x4,y4]]
        """
        diagBoxes =[[int(x[0]), int(x[1]), int(x[4]), int(x[5])] for x in ceilbox]

        self.diagBoxes = diagBoxes
        self.interval = interval
        self.batch()

    def batch(self):
        self.cor = []
        rowcor = self.table_line_cor(self.diagBoxes, axis='row', interval=self.interval)
        colcor = self.table_line_cor(self.diagBoxes, axis='col', interval=self.interval)
        cor = [{'row': line[1], 'col': line[0]} for line in zip(rowcor, colcor)]
        self.cor = cor

    def table_line_cor(self, lines, axis='col', interval=10):

        if axis == 'col':
            edges = [[line[1], line[3]] for line in lines]
        else:
            edges = [[line[0], line[2]] for line in lines]

        edges = sum(edges, [])
        edges = sorted(edges)

        nedges = len(edges)
        edgesMap = {}
        for i in range(nedges):
            if i == 0:
                edgesMap[edges[i]] = edges[i]
                continue
            else:
                if edges[i] - edgesMap[edges[i - 1]] < interval:
                    edgesMap[edges[i]] = edgesMap[edges[i - 1]]
                else:
                    edgesMap[edges[i]] = edges[i]

        edgesMapList = [[key, edgesMap[key]] for key in edgesMap]
        edgesMapIndex = [line[1] for line in edgesMapList]
        edgesMapIndex = list(set(edgesMapIndex))
        edgesMapIndex = {x: ind for ind, x in enumerate(sorted(edgesMapIndex))}

        if axis == 'col':
            cor = [[edgesMapIndex[edgesMap[line[1]]], edgesMapIndex[edgesMap[line[3]]]] for line in lines]
        else:
            cor = [[edgesMapIndex[edgesMap[line[0]]], edgesMapIndex[edgesMap[line[2]]]] for line in lines]
        return cor

import xlwt
def to_excel(res, workbook=None):
    ##res:[{'text': '购 买 方', 'cx': 192.0, 'w': 58.0, 'h': 169.0, 'cy': 325.5, 'angle': 0.0, 'row': [0, 1], 'col': [0, 1]}]
    row = 0
    if workbook is None:
        workbook = xlwt.Workbook()
    if len(res) == 0:
        worksheet = workbook.add_sheet('table')
        worksheet.write_merge(0, 0, 0, 0, "无数据")
    else:
        worksheet = workbook.add_sheet('page')
        pageRow = 0
        for line in res:
                row0, row1 = line['row']
                col0, col1 = line['col']
                text = line.get('text','')
                try:
                    pageRow = max(row1 - 1, pageRow)
                    worksheet.write_merge(row + row0, row + row1 - 1, col0, col1 - 1, text)
                except:
                    pass
    return workbook


if __name__=='__main__':
    pass
