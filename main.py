import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# 给定经纬度loc1，loc2, 返回两点间距离
# 单位：km
def get_distance(loc1, loc2):
    # 地球半径
    R = 6371
    # 经纬度转换成弧度
    loc1 = np.radians(loc1)
    loc2 = np.radians(loc2)
    # 计算经纬度差值
    dlon = loc2[0] - loc1[0]
    dlat = loc2[1] - loc1[1]
    # 计算距离
    a = np.sin(dlat / 2) ** 2 + np.cos(loc1[1]) * np.cos(loc2[1]) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    return distance

# 建一个nodeLoc字典
def build_nodeLoc(node):
    nodeLoc = {}
    for i in range(len(node)):
        # 将node.loc[i, 'loc']按逗号分离，分别转换成float类型
        nodeLoc[i + 1] = [float(x) for x in node.loc[i, 'loc'].split(',')]
    return nodeLoc

# 建立车速list
# 格式：{1：{2：40，6：40},...}
# 单位：km/h
def build_speed_list(road):
    car_speed = {}
    # 正向
    for i in range(len(road)):
        # 提取出od[i]中以逗号分隔的两个数字，分别赋值给x和y
        x, y = road.loc[i, 'od'].split(',')
        x = int(x)
        y = int(y)
        # 如果x不在car_speed中，就新建一个字典
        if x not in car_speed:
            car_speed[x] = {}
        # 如果y不在car_speed[x]中，就新建一个字典
        if y not in car_speed[x]:
            car_speed[x][y] = {}
        # 将速度赋值给car_speed[x][y]
        car_speed[x][y] = road.loc[i, 'v']
    # 反向
    for i in range(len(road)):
        # 提取出od[i]中以逗号分隔的两个数字，分别赋值给x和y
        y, x = road.loc[i, 'od'].split(',')
        x = int(x)
        y = int(y)
        # 如果x不在car_speed中，就新建一个字典
        if x not in car_speed:
            car_speed[x] = {}
        # 如果y不在car_speed[x]中，就新建一个字典
        if y not in car_speed[x]:
            car_speed[x][y] = {}
        # 将速度赋值给car_speed[x][y]
        car_speed[x][y] = road.loc[i, 'v']

    return car_speed

# 建立道路长度list
# 格式：{1：{2：1005，6：12456},...}
# 单位：m
def build_length_list(road):
    road_length = {}
    for i in range(len(road)):
        # 提取出od[i]中以逗号分隔的两个数字，分别赋值给x和y
        x, y = road.loc[i, 'od'].split(',')
        x = int(x)
        y = int(y)
        # 如果x不在road_length中，就新建一个字典
        if x not in road_length:
            road_length[x] = {}
        # 如果y不在road_length[x]中，就新建一个字典
        if y not in road_length[x]:
            road_length[x][y] = {}
        # 将速度赋值给road_length[x][y]
        road_length[x][y] = road.loc[i, 'len']

    for i in range(len(road)):
        # 提取出od[i]中以逗号分隔的两个数字，分别赋值给x和y
        y, x = road.loc[i, 'od'].split(',')
        x = int(x)
        y = int(y)
        # 如果x不在road_length中，就新建一个字典
        if x not in road_length:
            road_length[x] = {}
        # 如果y不在road_length[x]中，就新建一个字典
        if y not in road_length[x]:
            road_length[x][y] = {}
        # 将速度赋值给road_length[x][y]
        road_length[x][y] = road.loc[i, 'len']

    return road_length

# dijkstra算法
def dijkstra(G, from_node, to_node):
    # 用于存储已经求出最短距离的节点，存储被锁住的点
    S = []
    # 用于存储最短距离
    dis = {}
    # 用于存储最短路径
    path = {}
    # 初始化dis和path
    for i in G:
        dis[i] = float('inf')
        path[i] = []
    dis[from_node] = 0
    path[from_node] = [from_node]
    while len(S) < len(G):
        # 从未求出最短距离的节点中找出距离最小的节点
        min_dis = float('inf')
        for i in dis:
            if i not in S and dis[i] < min_dis:
                min_dis = dis[i]
                u = i
        S.append(u)
        # 更新dis和path
        for i in G[u]:
            if (dis[u] + G[u][i] < dis[i]) & (i not in S):
                dis[i] = dis[u] + G[u][i]
                path[i] = path[u] + [i]

    # 如果S中没有to_node，说明无法到达to_node
    if to_node not in S:
        return None, None

    # 这里只返回to_node的最短距离和最短路径, 其实也可以返回所有点的dis和path
    return dis[to_node], path[to_node]

# 三个点提供经纬度，算夹角
def get_angle(loc1, loc2, loc3):
    def geo2xyz(lat, lng, r=6371):
        '''
        将地理经纬度转换成笛卡尔坐标系
        :param lat: 纬度
        :param lng: 经度
        :param r: 地球半径
        :return: 返回笛卡尔坐标系
        '''
        thera = (math.pi * lat) / 180
        fie = (math.pi * lng) / 180
        x = r * math.cos(thera) * math.cos(fie)
        y = r * math.cos(thera) * math.sin(fie)
        z = r * math.sin(thera)
        return [x, y, z]

    p1 = geo2xyz(loc1[0], loc1[1])
    p2 = geo2xyz(loc2[0], loc2[1])
    p3 = geo2xyz(loc3[0], loc3[1])

    _P1P2 = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)
    _P2P3 = math.sqrt((p3[0] - p2[0]) ** 2 + (p3[1] - p2[1]) ** 2 + (p3[2] - p2[2]) ** 2)
    P = (p1[0] - p2[0]) * (p3[0] - p2[0]) + (p1[1] - p2[1]) * (p3[1] - p2[1]) + (p1[2] - p2[2]) * (p3[2] - p2[2])
    angle = (math.acos(P / (_P1P2 * _P2P3)) / math.pi) * 180
    return angle

# 优化h()函数，加入方向性
def betterH(car_speed, road_length, nodeLoc, to_node, next_node, now_node, w = 10):
    # 计算A，如果now_node,next_node,to_node的夹角<90,则Anexti=w(1+cos(theta)),否则A=1
    if get_angle(nodeLoc[now_node], nodeLoc[next_node], nodeLoc[to_node]) < 90:
        A = w * (1 + math.cos(get_angle(nodeLoc[now_node], nodeLoc[next_node], nodeLoc[to_node])))
    else:
        A = 1
    # 后续点到终点的距离 单位km
    d = get_distance(nodeLoc[next_node], nodeLoc[to_node])
    H = 1.0/car_speed[now_node][next_node]*(d * A)
    H = H*60/1000
    # print(H)
    return H

# TODO
# A-star 算法
# 输入：路网拓扑G, 起点from_node, 终点to_node
# 输出：最优行驶路径
def A_star(G, from_node, to_node, nodeLoc, car_speed, road_length):
    # 初始化open表和close表
    open = []
    close = []
    # 将起点放入open表
    open.append(from_node)
    # 初始化g(n)和h(n)
    g = {}
    h = {}
    # 初始化f(n)
    f = {}
    # 初始化父节点
    parent = {}
    # 初始化g(n)
    g[from_node] = 0
    # 初始化h(n)
    h[from_node] = get_distance(nodeLoc[from_node], nodeLoc[to_node])
    # 初始化f(n)
    f[from_node] = g[from_node] + h[from_node]
    # 初始化父节点
    parent[from_node] = None
    # 当open表不为空时
    while open:
        # 从open表中找出f(n)最小的节点
        min_f = float('inf')
        for i in open:
            if f[i] < min_f:
                min_f = f[i]
                u = i
        # 将u从open表中删除
        open.remove(u)
        # 将u放入close表
        close.append(u)
        # 如果u是终点，返回路径
        if u == to_node:
            path = []
            while u:
                path.append(u)
                u = parent[u]
            return path[::-1]
        # 对u的所有邻居进行处理
        for v in G[u]:
            # 如果v在close表中，跳过
            if v in close:
                continue
            # 如果v不在open表中，加入open表
            if v not in open:
                open.append(v)
            # 如果g[v]不存在，那么g[v] = inf
            if v not in g:
                g[v] = float('inf')
            # 计算g(n)
            if (g[u] + G[u][v] < g[v]) & (v not in close):
                g[v] = g[u] + G[u][v]
                # 更新父节点
                parent[v] = u
            # 计算h(n)
            if v == to_node:
                # h[v] = G[u][v]
                h[v] = 0
            else:
                h[v] = betterH(car_speed, road_length, nodeLoc, to_node, now_node=u, next_node=v)
            # 计算f(n)
            f[v] = g[v] + h[v]
    # 如果open表为空，说明没有找到路径
    return None



if __name__ == '__main__':
    # 从node.xlsx中读取节点信息，读入到字典node中
    ##########################################
    # 节点编号node_id 经纬度loc(Lon, Lat)   直连道路编号dirc_road  状态state
    # 1             118.737492,32.069828  1, 2             0
    # 2             118.791462,32.064427  1, 3             0
    # 3             118.819058,32.063448  2, 4             0
    # 4             118.771771,32.047779  3                0
    ##########################################
    node = pd.read_excel('node.xlsx')

    # 从road.xlsx中读取道路信息，读入到字典road中
    # 读入道路
    ###########################################################################################################################
    # 道路编号road_id 道路实际名称name 边界节点编号od(起始点, 终止点) 道路长度len(m) 车道数量c 平均车速v(km/h) 预测步长s 道路时空拥挤系数(通行时间)w(min) 道路不稳定时间区间集合T
    # 1                 road1         (1, 2)                    1            2          60          1           1                       (1,2,3,4,5,6,7,10)
    # 2                 road2         (1, 3)                    1            1          40          1           3                       (1,2,3,4,5,,10)
    # 3                 road3         (2, 4)                    1            4          63          1           3                       (1,2,3,4,7,10)
    # 4                 road4         (3, 4)                    1            1          62          1           1                       (1,4,5,6,7)
    ###########################################################################################################################
    road = pd.read_excel('road.xlsx')

    # # 提取node中的经纬度信息
    # loc = node['loc'].values
    #
    # # 读取road中的边界节点编号od，从node中提取对应的经纬度信息，计算两点间距离
    # od = road['od'].values
    #
    # for i in range(len(od)):
    #     # 提取出od[i]中以逗号分隔的两个数字，分别赋值给x和y
    #     x, y = od[i].split(',')
    #     x = int(x)
    #     y = int(y)
    #
    #     loc1 = loc[x - 1]
    #     loc2 = loc[y - 1]
    #     print(loc1, loc2)
    #
    #     # 将经纬度字符串转换成浮点数
    #     loc1 = [float(i) for i in loc1.split(',')]
    #     loc2 = [float(i) for i in loc2.split(',')]
    #     print(loc1, loc2)
    #     distance = get_distance(loc1, loc2) # 单位km
    #
    #     # 所得distance*1000后加[50,250]均匀采样，得到道路长度
    #     road.loc[i, 'len'] = int(np.random.uniform(distance*1000+50, distance*1000+250))
    #
    #     # 将len列写入到road.xlsx中，覆盖原来的len列
    #     road.to_excel('road.xlsx', index=False)

    # 根据rode的信息，绘制道路图，其中road_id为道路编号，od为道路的起始节点和终止节点，len为道路长度

    # 根据road构建图，这里先默认是双向图
    # graph = {1: {2: 7.8645, 6: 6.447, 10: 7.3095},2: {1: 7.8645, 3: 4.116, 6: 4.2315000000000005}, ...}
    graph = {}
    for i in range(len(road)):
        # 提取出od[i]中以逗号分隔的两个数字，分别赋值给x和y
        x, y = road.loc[i, 'od'].split(',')
        x = int(x)
        y = int(y)
        # 将道路信息存入graph中
        if x not in graph:
            graph[x] = {}
        if y not in graph:
            graph[y] = {}
        graph[x][y] = road.loc[i, 'len']/road.loc[i, 'v']/1000*60
        graph[y][x] = road.loc[i, 'len']/road.loc[i, 'v']/1000*60
    print('Graph:', graph)

    # 建立nodeLoc字典，存储节点编号和经纬度信息
    nodeLoc = build_nodeLoc(node)
    print('nodeLoc:', nodeLoc)

    car_speed = build_speed_list(road)
    print('car_speed:', car_speed)

    road_length = build_length_list(road)
    print('road_length', road_length)

    # A-star算法求1-24的最短路径
    path_A_star = A_star(graph, 1, 24, nodeLoc, car_speed, road_length)
    print('A-star path:', path_A_star)

    # dijkstra算法求1-24的最短路径
    dis_dij, path_dij = dijkstra(graph, 1, 24)
    print('dijkstra path:', path_dij)
















