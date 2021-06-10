---
layout: post
title:  "Q & A"
---
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {
    inlineMath: [['$','$'], ['\\(','\\)']],
    processEscapes: true
  }
});
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


{:toc}

* 
{:toc}


<style>
table {
  border-collapse: collapse;
  border: 1px solid black;
  margin: 0 auto;
} 

th,td {
  border: 1px solid black;
  text-align: center;
  padding: 20px;
}

table.a {
  table-layout: auto;
  width: 180px;  
}

table.b {
  table-layout: fixed;
  width: 600px;  
}

table.c {
  table-layout: auto;
  width: 100%;  
}

table.d {
  table-layout: fixed;
  width: 100%;  
}
</style>


## 1. C++
<p align="justify">
<b>How to new a 2D matrix?</b><br>
</p>
{% highlight C++ %}
int n = 5;
int **g = new int *[n];
for (int i = 0; i < n; i++)
{
    g[i] = new int[n];
}
for (int i = 0; i < n; i++)
{
    delete []g[i];
}
delete []g;
{% endhighlight %}

<p align="justify">
<b>How to convert a char to string like 'a' to "a"?</b><br>
</p>
{% highlight C++ %}
string res = "";
res.push_back('a');  // res = "a"
res += 'b';  // res = "ab"
res.append(1, 'c');  // res = "abc"
res.assign(1, 'd');  // res = "d";
res.insert(1, 1, 'e');  // res = "de"
res.insert(0, 1, 'e');  // res = "ed"
res.replace(0, 1, 1, 'f');  // res = "fe"
string res(2, 'a');  // res = "aa"
{% endhighlight %}

<p align="justify">
<b>How to print an inversed string in a recursive way?</b><br>
</p>
{% highlight C++ %}
void printInverseStrRecur(string str, int idx, int len)
{
    if (idx < len)
    {
        printInverseStrRecur(str, idx+1, len);
        cout << str[idx];
    }
}
{% endhighlight %}

<p align="justify">
二项分布，n次试验，每次试验成功的概率为p，求问连续k次成功的概率为多少？<br>
例如：<br>
input: n = 2, k = 1, p = 0.5<br>
output: 0.75<br>
input: n = 10, k = 3, p = 0.4<br>
output: 0.3141
</p>
{% highlight C++ %}
void calProb(bool *dp, int n, int k, float p, int idx, float &res)
{
    if (idx == n)
    {
        for (int i = 0; i < n; i++) { printf("%d\t", dp[i]); }
        printf("\n");
        bool isOk = false;
        for (int i = 0; i < n-k+1; i++)
        {
            bool isK = true;
            for (int j = i; j < i+k; j++)
            {
                if (!dp[j])
                {
                    isK = false;
                    break;
                }
            }
            if (isK)
            {
                isOk = true;
                break;
            }
        }
        if (isOk)
        {
            float prob = 1.0;
            for (int i = 0; i < n; i++)
            {
                if (dp[i]) { prob *= p; }
                else { prob *= (1-p); }
            }
            printf("%f\n", prob);
            res += prob;
        }
    }
    else
    {
        dp[idx] = true;
        path(dp, n, k, p, idx+1, res);
        dp[idx] = false;
        path(dp, n, k, p, idx+1, res);
    }
}

int amin(int argc, const char * argv[])
{
    int n = 10, k = 3;
    float p = 0.4, res = 0.0;
    scanf("%d%d%f", &n, &k, &p);
    bool *dp = new bool [n]{};
    calProb(dp, n, k, p, 0, res);
    delete []dp;
    printf("%.4f\n", res);
    return 0;
}
{% endhighlight %}

<p align="justify">
We have 4 numbers. Apply four basic operator (+, -, *, /) to get 24.
</p>
{% highlight C++ %}
#include<iostream>
#include<string>
#include<vector>
#include<stack>
#include<queue>
#include<cmath>
#include<iomanip>
using namespace std;


class Solution {
public:
    /**
     *
     * @param arr int整型一维数组
     * @param arrLen int arr数组长度
     * @return bool布尔型
     */
    bool Game24Points(int* arr, int arrLen) {
        // write code here
        int n = arrLen;
        if (n != 4) { return false; }
        vector<vector<int>> all;
        permute(arr, all, n, 0);
        /*
        for (int i = 0; i < int(all.size()); i++)
        {
            for (int j = 0; j < int(all[i].size()); j++)
            {
                cout << all[i][j] << '\t';
            }
            cout << endl;
        }
        */
        vector<string> allOpt = allOperators();
        for (int i = 0; i < int(all.size()); i++)
        {
            int *a = new int [4]{};
            a[0] = all[i][0];
            a[1] = all[i][1];
            a[2] = all[i][2];
            a[3] = all[i][3];
            for (int j = 0; j < int(allOpt.size()); j++)
            {
                if (isOK(a, allOpt[j], n))
                {
                    return true;
                }
            }
            delete []a;
        }
        return false;
    }
    vector<string> allOperators()
    {
        vector<string> res;
        string opt = "+-*/";
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                for (int k = 0; k < 4; k++)
                {
                    string temp = "";
                    temp.push_back(opt[i]);
                    temp.push_back(opt[j]);
                    temp.push_back(opt[k]);
                    res.push_back(temp);
                }
            }
        }
        return res;
    }
    bool isOK(int *arr, string opt, int n)
    {
        stack<int> ns;
        stack<char> os;
        ns.push(arr[0]);
        int temp = 0;
        for (int i = 0; i < 3; i++)
        {
            if (opt[i] == '*')
            {
                temp = ns.top();
                ns.pop();
                ns.push(temp * arr[i+1]);
            }
            else if (opt[i] == '/')
            {
                if (ns.top() % arr[i+1] ) { return false; }
                temp = ns.top();
                ns.pop();
                ns.push(temp / arr[i+1]);
            }
            else
            {
                ns.push(arr[i+1]);
                os.push(opt[i]);
            }
        }
        stack<int> nns;
        stack<char> nos;
        while (!ns.empty())
        {
            nns.push(ns.top());
            ns.pop();
        }
        while (!os.empty())
        {
            nos.push(os.top());
            os.pop();
        }
        while (!nos.empty())
        {
            int a = nns.top();
            nns.pop();
            int b = nns.top();
            nns.pop();
            if (nos.top() == '+') { temp = a + b; }
            else if (nos.top() == '-') { temp = a - b; }
            else if (nos.top() == '*') { temp = a * b; }
            else
            {
                if (a % b) { return false; }
                temp = a / b;
            }
            nos.pop();
            nns.push(temp);
        }
        cout << nns.top() << endl;
        return nns.top() == 24;
    }
    void permute(int *arr, vector<vector<int>> &all, int n, int idx)
    {
        if (idx == n-1)
        {
            vector<int> temp;
            for (int i = 0; i < n; i++)
            {
                temp.push_back(arr[i]);
            }
            all.push_back(temp);
        }
        else
        {
            for (int i = idx; i < n; i++)
            {
                swap(arr, idx, i);
                permute(arr, all, n, idx+1);
                swap(arr, idx, i);
            }
        }
    }
    void swap(int *arr, int i, int j)
    {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
};


int main(int argc, const char * argv[]) {
    // insert code here...
    Solution sol;
    int *arr = new int [4]{};
    arr[0] = 7;
    arr[1] = 2;
    arr[2] = 1;
    arr[3] = 10;
    //cout << sol.isOK(arr, "*++", 4) << endl;
    //cout << sol.Game24Points(arr, 4) << endl;
    cout << sol.isOK(arr, "---", 4) << endl;
    return 0;
}
{% endhighlight %}

<p align="justify">
李白有2斗酒，出门遇到酒店话，酒加倍，遇到莲花话，喝酒一斗。问他遇到5次酒店，10次莲花，且最后一次是莲花，刚好把酒喝完，这样的莲花、酒店组成的序列有多少种？
</p>
{% highlight C++ %}
void move(int f, int h, int nF, int nH, int &count, int res)
{
    //cout << nF << '\t' << nH << endl;
    if (f == nF && h == nH)
    {
        if (res == 1)
        {
            count++;
        }
        return;
    }
    if (nF > f || nH > h || res < 0) { return; }
    move(f, h, nF+1, nH, count, res-1);
    move(f, h, nF, nH+1, count, res*2);
}


int main(int argc, const char * argv[]) {
    // insert code here...
    int f = 9;
    int h = 5;
    int res = 2;
    int count = 0;
    move(f, h, 0, 0, count, res);
    cout << count << endl;
    return 0;
}
{% endhighlight %}

<p align="justify">
石头<br>
题目描述：<br>
沙滩按照线型摆放着n个大小不一的球形石头，已知第i个石头的半径为ri，且不存在两个石头有相同的半径。为了使石头的摆放更加美观，现要求摆放的石头的半径从左往右依次递增。因此，需要对一些石头进行移动，每次操作可以选择一个石头，并把它放在剩下n−1个石头在最左边或最右边。问最少需要操作多少次才能将这n个石头的半径变成升序？<br><br>

输入描述<br>
第一行一个整数n，表示石头的个数。(1 <= n <= 100000) 第二行n个整数，表示从左往右石头的半径r1，r2，...，rn。(1 <= ri <= n)，且保证不存在两个不同的石头拥有相同的半径。<br><br>

输出描述<br>
最少操作次数<br><br>

样例输入<br>
5<br>
4 1 2 5 3<br>
样例输出<br>
2
</p>
{% highlight C++ %}
void stone()
{
    int n = 5, maxLen = 1;
    cin >> n;
    int *arr = new int [n]{};
    for (int i = 0; i < n; i++) { cin >> arr[i]; }
    unordered_map<int, int> dict;
    unordered_map<int, int>::iterator iter;
    for (int i = 0; i < n; i++)
    {
        dict[arr[i]] = 1;
        iter = dict.find(arr[i]-1);
        if (iter != dict.end())
        {
            dict[arr[i]] = dict[arr[i]-1] + 1;
            if (dict[arr[i]] > maxLen) { maxLen = dict[arr[i]]; }
        }
    }
    delete []arr;
    printf("%d\n", n - maxLen);
}
{% endhighlight %}

<p align="justify">
01交替<br>
题目描述：<br>
给你一个长度为n的01串。现在想让你找出最长的01交替子串（子串可以不连续）比如：1010,0101是01交替的串，1101则不是。现在你可以把某一个连续的区间进行翻转，即0变1，1变0。问修改之后的最大01交替子串的长度是多少。<br><br>

输入描述<br>
输入第一行包含一个整数n （1 <= n <= 100000) 表示01串的长度。 第二行包含一个01串。<br><br>

输出描述<br>
输出一个数表示符合题意的子串长度<br><br>

样例输入<br>
8<br>
10000011<br>
样例输出<br>
5
</p>
{% highlight C++ %}
void count01()
{
    int n = 8;
    cin >> n;
    string str = "";
    cin >> str;
    int count = 1;
    for (int i = 0; i < n-1; i++)
    {
        if (str[i] != str[i+1]) { count++; }
    }
    if (count < n-1) { printf("%d\n", count+2); }
    else { printf("%d\n", n); }
}
{% endhighlight %}

<p align="justify">
25匹马，5个赛道，每次只能进行一场比赛，求问多少场比赛可以找出前三名。<br>
7场<br>
1、将25匹马分成5组，每组进行一场比赛，得到排名，假设排名如下：<br>
A1 > A2 > A3 > A4 > A5<br>
B1 > B2 > B3 > B4 > B5<br>
C1 > C2 > C3 > C4 > C5<br>
D1 > D2 > D3 > D4 > D5<br>
E1 > E2 > E3 > E4 > E5<br>
2、将每组的第一名进行一轮比赛，假设排名如下<br>
A1 > B1 > C1 > D1 > E1<br>
D组和E组可以舍弃，因为D1和E1不能进入前三<br>
有望争夺前三的是A2, A3, B1, B2, C1<br>
3、上述的5匹马进行一轮比赛，得到前两名
</p>

<p align="justify">
new和malloc的区别<br>
1、new是操作符，而malloc是函数<br>
2、new在调用的时候先分配内存，在调用构造函数，释放的时候调用析构函数。<br>
3、new是类型安全的，malloc返回void*<br>
4、new可以被重载<br>
5、malloc 可以被realloc<br>
6、new发生错误抛出异常，malloc返回null<br>
7、malloc可以分配任意字节，new 只能分配实例所占内存的整数倍数大小
</p>

## 计算机操作系统
<p align="justify">

</p>

<p align="justify">
<a href="https://zhuanlan.zhihu.com/p/46368084"> Python里面有多线程吗？</a><br>
由于GIL的存在，很多人认为Python多进程编程更快，针对多核CPU，理论上来说也是采用多进程更能有效利用资源。<br>
对CPU密集型代码(比如循环计算) - 多进程效率更高<br>
对IO密集型代码(比如文件操作，网络爬虫) - 多线程效率更高。<br>
为什么是这样呢？其实也不难理解。对于IO密集型操作，大部分消耗时间其实是等待时间，在等待时间中CPU是不需要工作的，那你在此期间提供双CPU资源也是利用不上的，相反对于CPU密集型代码，2个CPU干活肯定比一个CPU快很多。那么为什么多线程会对IO密集型代码有用呢？这时因为python碰到等待会释放GIL供新的线程使用，实现了线程间的切换。<br><br>

<a href="https://www.zhihu.com/question/23474039/answer/269526476"> 摘录于此</a><br>
第一种任务的类型是计算密集型任务的特点是要进行大量的计算，消耗CPU资源，比如计算圆周率、对视频进行高清解码等等，全靠CPU的运算能力。这种计算密集型任务虽然也可以用多任务完成，但是任务越多，花在任务切换的时间就越多，CPU执行任务的效率就越低，所以，要最高效地利用CPU，计算密集型任务同时进行的数量应当等于CPU的核心数。计算密集型任务由于主要消耗CPU资源，因此，代码运行效率至关重要。Python这样的脚本语言运行效率很低，完全不适合计算密集型任务。对于计算密集型任务，最好用C语言编写。<br>
第二种任务的类型是IO密集型，涉及到网络、磁盘IO的任务都是IO密集型任务，这类任务的特点是CPU消耗很少，任务的大部分时间都在等待IO操作完成（因为IO的速度远远低于CPU和内存的速度）。对于IO密集型任务，任务越多，CPU效率越高，但也有一个限度。常见的大部分任务都是IO密集型任务，比如Web应用。IO密集型任务执行期间，99%的时间都花在IO上，花在CPU上的时间很少，因此，用运行速度极快的C语言替换用Python这样运行速度极低的脚本语言，完全无法提升运行效率。对于IO密集型任务，最合适的语言就是开发效率最高（代码量最少）的语言，脚本语言是首选，C语言最差。<br>
综上，Python多线程相当于单核多线程，多线程有两个好处：CPU并行，IO并行，单核多线程相当于自断一臂。所以，在Python中，可以使用多线程，但不要指望能有效利用多核。如果一定要通过多线程利用多核，那只能通过C扩展来实现，不过这样就失去了Python简单易用的特点。不过，也不用过于担心，Python虽然不能利用多线程实现多核任务，但可以通过多进程实现多核任务。多个Python进程有各自独立的GIL锁，互不影响。
</p>

<p align="justify">
GBDT与随机森林有什么不同点？<br>
DT + Boosting = GBDT<br>
GBDT是一种boosting算法。boosting工作机制：先从初始训练集训练处一个基学习器，然后在根据基学习器的表现对训练样本分布进行调整，使得先前的基学习器做错的训练样本在后续获得更多关注（增加错误样本权重），然后基于调整后的样本分布训练下一个基学习器，如此重复，直到基学习器达到指定的T时，最终将T个基学习器进行加权结合，得出预测。<br>
DT + Bagging = RF<br>
随机森林是bagging的一种扩展，在k个数据集选择的时候后，引入了随机属性选择。加入所有属性个数为d，k是随机选择的属性个数。那么k=d的时候，就没有改变。那么k=1的时候后，随机选择一个属性用于计算。推荐的k=log2d.<br>
随机森林的基学习器一般是决策树算法-主要，也有神经网络。<br>
随机森林是对bagging算法的一点改动，但是根能提现样本集之间的差异性。会提高最终预测结果的泛化能力。<br><br>

GBDT和随机森林的相同点<br>
1、都是由多棵树组成<br>
2、最终的结果都是由多棵树一起决定<br><br>

GBDT和随机森林的不同点<br>
1、组成随机森林的树可以是分类树，也可以是回归树；而GBDT只由回归树组成<br>
2、组成随机森林的树可以并行生成；而GBDT只能是串行生成<br>
3、对于最终的输出结果而言，随机森林采用多数投票等；而GBDT则是将所有结果累加起来，或者加权累加起来<br>
4、随机森林对异常值不敏感，GBDT对异常值非常敏感<br>
5、随机森林对训练集一视同仁，GBDT是基于权值的弱分类器的集成<br>
6、随机森林是通过减少模型方差提高性能，GBDT是通过减少模型偏差提高性能<br><br>

Bagging（套袋法）的算法过程如下：<br>
从原始样本集中使用Bootstraping方法随机抽取n个训练样本，共进行k轮抽取，得到k个训练集。（k个训练集之间相互独立，元素可以有重复）<br>
对于k个训练集，我们训练k个模型（这k个模型可以根据具体问题而定，比如决策树，knn等）<br>
对于分类问题：由投票表决产生分类结果；对于回归问题：由k个模型预测结果的均值作为最后预测结果。（所有模型的重要性相同）<br><br>

Boosting（提升法）的算法过程如下：<br>
对于训练集中的每个样本建立权值wi，表示对每个样本的关注度。当某个样本被误分类的概率很高时，需要加大对该样本的权值。<br>
进行迭代的过程中，每一步迭代都是一个弱分类器。我们需要用某种策略将其组合，作为最终模型。（例如AdaBoost给每个弱分类器一个权值，将其线性组合最为最终分类器。误差越小的弱分类器，权值越大）<br><br>

Bagging，Boosting的主要区别<br>
样本选择上：Bagging采用的是Bootstrap随机有放回抽样；而Boosting每一轮的训练集是不变的，改变的只是每一个样本的权重。<br>
样本权重：Bagging使用的是均匀取样，每个样本权重相等；Boosting根据错误率调整样本权重，错误率越大的样本权重越大。<br>
预测函数：Bagging所有的预测函数的权重相等；Boosting中误差越小的预测函数其权重越大。<br>
并行计算：Bagging各个预测函数可以并行生成；Boosting各个预测函数必须按顺序迭代生成。<br><br>
</p>

<p align="justify">
SVM可以做回归问题吗？<br>
<a href="https://zhuanlan.zhihu.com/p/33692660"> 摘录于此</a><br>
最简单的线性回归模型是要找出一条曲线使得残差最小。同样的，SVR也是要找出一个超平面，使得所有数据到这个超平面的距离最小。SVR是SVM的一种运用，基本的思路是一致，除了一些细微的区别。使用SVR作回归分析，与SVM一样，我们需要找到一个超平面，不同的是：在SVM中我们要找出一个间隔（gap）最大的超平面，而在SVR，我们定义一个ε，如上图所示，定义虚线内区域的数据点的残差为0，而虚线区域外的数据点（支持向量）到虚线的边界的距离为残差（ζ）。与线性模型类似，我们希望这些残差（ζ）最小。所以大致上来说，SVR就是要找出一个最佳的条状区域（2ε宽度），再对区域外的点进行回归。<br><br>
</p>

<p align="justify">
Python里面，为什么要用if __name__ == '__main__'
</p>

<p align="justify">
稳定排序有那些？<br>
归并、冒泡、插入、桶排
</p>

<p align="justify">
对输入样本故意添加一些人无法察觉的细微的干扰，这种样本叫什么？<br>
对抗样本(Adversarial Examples)
</p>

<p align="justify">
正向最大分词和逆向最大分词
</p>

<p align="justify">
平均抛硬币多少次可以得到连续两个正面？<br>
三个状态：0个H（初始状态）、1个H、2个H。状态转移方程
$$
\begin{pmatrix}
0.5 & 0.5 & 0\\
0.5 & 0 & 0.5\\
0 & 0 & 1
\end{pmatrix}
$$

设$k_{i}^{A}$从状态i到状态A的平均步数
$$
k_{i}^{A} =
\begin{cases}
0, & \quad i \in A\\
1 + \sum_{j \notin A} P_{ij} k_{j}^{A}, & \quad i \notin A
\end{cases}
$$

$$
\begin{aligned}
& k_{1}^{3} = 1 + P_{11} k_{1}^{3} + P_{12} k_{2}^{3}\\
& k_{2}^{3} = 1 + P_{21} k_{1}^{3} + P_{22} k_{2}^{3}\\
& k_{3}^{3} = 0
\end{aligned}
$$

解得
$$
\begin{aligned}
& k_{1}^{3} = 6\\
& k_{2}^{3} = 4\\
& k_{3}^{3} = 0
\end{aligned}
$$
</p>

<p align="justify">
如何处理正负样本不均匀的情况？
</p>

<p align="justify">
多多的魔术盒子<br>
多多鸡有N个魔术盒子（编号1～N），其中编号为i的盒子里有i个球。<br>
多多鸡让皮皮虾每次选择一个数字X（1 <= X <= N），多多鸡就会把球数量大于等于X个的盒子里的球减少X个。<br>
通过观察，皮皮虾已经掌握了其中的奥秘，并且发现只要通过一定的操作顺序，可以用最少的次数将所有盒子里的球变没。<br>
那么请问聪明的你，是否已经知道了应该如何操作呢？<br><br>

<b>输入描述:</b><br>
第一行，有1个整数T，表示测试用例的组数。<br>
（1 <= T <= 100）<br>
接下来T行，每行1个整数N，表示有N个魔术盒子。<br>
（1 <= N <= 1,000,000,000）<br><br>

<b>输出描述:</b><br>
共T行，每行1个整数，表示要将所有盒子的球变没，最少需要进行多少次操作。<br><br>

<b>示例:</b><br>
输入<br>
3<br>
1<br>
2<br>
5<br>
输出<br>
1<br>
2<br>
3<br><br>

<b>Solution:</b>
$$
f(x) = 
\begin{cases}
1, &\quad x = 1 \\
1 + f(x - \left \lfloor \frac{1 + x}{2} \right \rfloor), &\quad x > 1
\end{cases}
$$
</p>
{% highlight C++ %}
#include<iostream>
using namespace std;

int getReduceTimes(int n)
{
    if (n == 1) { return 1; }
    else { return 1 + getReduceTimes(n-(1+n)/2); }
}

int main(int argc, const char * argv[])
{
    int T = 3, n = 1;
    scanf("%d", &T);
    while (T--)
    {
        scanf("%d", &n);
        printf("%d\n", getReduceTimes(n));
    }
    return 0;
}
{% endhighlight %}

<p align="justify">
多多的排列函数<br>
数列 {An} 为N的一种排列。例如N=3，可能的排列共6种：<br>
1, 2, 3<br>
1, 3, 2<br>
2, 1, 3<br>
2, 3, 1<br>
3, 1, 2<br>
3, 2, 1<br>
定义函数F:
$$
F(x) =
\begin{cases}
A_{1},  &\quad x = 1\\
\left | F(x-1) - A_{x} \right |, &\quad x > 1
\end{cases}
$$
其中|X|表示X的绝对值。现在多多鸡想知道，在所有可能的数列 {An} 中，F(N)的最小值和最大值分别是多少。<br><br>

<b>输入描述:</b><br>
第一行输入1个整数T，表示测试用例的组数。<br>
( 1 <= T <= 10 )<br>
第二行开始，共T行，每行包含1个整数N，表示数列 {An} 的元素个数。<br>
( 1 <= N <= 100,000 )<br><br>

<b>输出描述:</b><br>
共T行，每行2个整数，分别表示F(N)最小值和最大值<br><br>

<b>示例:</b><br>
输入<br>
2<br>
2<br>
3<br>
输出<br>
1 1<br>
0 2<br>
说明<br>
对于N=3：<br>
- 当{An}为3，2，1时可以得到F(N)的最小值0<br>
- 当{An}为2，1，3时可以得到F(N)的最大值2<br><br>

<b>备注:</b><br>
对于60%的数据有： 1 <= N <= 100<br>
对于100%的数据有：1 <= N <= 100,000<br><br>

<b>Solution:</b><br>
1.最小值看n%4的余数，余数为1或2时，取值为1，其他为0；<br>
2.最大值相当于在1,2,...,n-1数列的一种最小值排列的右边再放一个n，所以最大值取值为Fmin-n；
</p>
{% highlight C++ %}
#include<iostream>
using namespace std;

int getMin(int n)
{
    if (n % 4 == 1 || n % 4 == 2) { return 1; }
    else { return 0; }
}

int getMax(int n)
{
    return n - getMin(n-1);
}

int main(int argc, const char * argv[])
{
    int T = 2, i = 1;
    scanf("%d", &T);
    while (T--)
    {
        scanf("%d", &i);
        printf("%d %d\n", getMin(i), getMax(i));
    }
    return 0;
}
{% endhighlight %}

<p align="justify">
多多的电子字典<br>
多多鸡打算造一本自己的电子字典，里面的所有单词都只由a和b组成。每个单词的组成里a的数量不能超过N个且b的数量不能超过M个。多多鸡的幸运数字是K，它打算把所有满足条件的单词里的字典序第K小的单词找出来，作为字典的封面。<br><br>

<b>输入描述:</b><br>
共一行，三个整数N, M, K。(0 < N, M < 50, 0 < K < 1,000,000,000,000,000)<br><br>

<b>输出描述:</b><br>
共一行，为字典序第K小的单词。<br><br>

<b>示例:</b><br>
输入<br>
2 1 4<br>
输出<br>
ab<br>
说明<br>
满足条件的单词里，按照字典序从小到大排列的结果是<br>
a<br>
aa<br>
aab<br>
ab<br>
aba<br>
b<br>
ba<br>
baa<br><br>

<b>备注:</b><br>
对于40%的数据：0 < K < 100,000<br>
对于100%的数据：0 < K < 1,000,000,000,000,000<br>
题目保证第K小的单词一定存在<br><br>

<b>Solution:</b><br>
本题相当于有两个树，以a开头和以b开头的两颗二叉树。 每一次的count是返回以当前节点"a"或者"b"开头的前序遍历需要经过的步数，这里说一下动态规划的递推公式：dp[i][j] = dp[i-1][j] + 1 + dp[i][j-1] + 1 什么意思呢？ 就是返回以"a"开头的和以"b"开头的个数。比如a[2][1] = dp[1][1] + dp[2][0] + 2 dp[1][1] + 1是以"a"开头的个数，即我们可以固定一个"a"在开头。这样就少了一个a可以用，那么只剩下了[a,b]可以组合 即：a,ab,ba,b; 然后再和固定的a开头的字母"a"组合，即aa,aab,aba,ab，这个时候组合就相当于是dp[1][1]。以a开头的组合显然少了一个"a"这个特殊的组合，显然dp[1][1] + 1才是以"a"开头的组合个数。
</p>
{% highlight C++ %}
#include<iostream>
#include<string>
using namespace std;

string dict(int n, int m, long k)
{
    string res = "a";
    int len = 1;
    unsigned long long **dp = new unsigned long long *[n+1];
    for (int i = 0; i <= n; i++)
    {
        dp[i] = new unsigned long long [m+1]{};
        dp[i][0] = i;
    }
    for (int j = 0; j <= m; j++) { dp[0][j] = j; }
    
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= m; j++)
        {
            dp[i][j] = dp[i-1][j] + dp[i][j-1] + 2;
        }
    }
    
    n--;
    k--;
    while (k > 0 && (n > 0 || m > 0))
    {
        unsigned long long count = dp[n][m] + 1;
        if (count <= k)
        {
            k -= count;
            n++;
            m--;
            res[len-1] = 'b';
        }
        else
        {
            k--;
            if (n > 0)
            {
                res.push_back('a');
                n--;
            }
            else
            {
                res.push_back('b');
                m--;
            }
            len++;
        }
    }
    for (int i = 0; i <= n; i++) { delete []dp[i]; }
    delete []dp;
    return res;
}

int main(int argc, const char *argv[])
{
    int n = 2, m = 1;
    long k = 4;
    scanf("%d%d%ld", &n, &m, &k);
    printf("%s\n", dict(n, m, k).c_str());
    return 0;
}
{% endhighlight %}

<p align="justify">
骰子期望<br>
扔n个骰子，第i个骰子有可能投掷出Xi种等概率的不同的结果，数字从1到Xi。所有骰子的结果的最大值将作为最终结果。求最终结果的期望。<br><br>

<b>输入描述:</b><br>
第一行一个整数n，表示有n个骰子。（1 <= n <= 50）<br>
第二行n个整数，表示每个骰子的结果数Xi。(2 <= Xi <= 50)<br><br>

<b>输出描述:</b><br>
输出最终结果的期望，保留两位小数。<br><br>

<b>示例:</b><br>
输入<br>
2<br>
2 2<br>
输出<br>
1.75<br><br>

<b>Solution:</b>
$$P(x = A) = P(x \leq A) - P(x \leq (A-1))$$
</p>
{% highlight C++ %}
#include<iostream>
using namespace std;

double calProb(int *arr, int n, int MAX)
{
    double res = 0.0, pre = 0.0, cur = 0.0;
    for (int i = 1; i <= MAX; i++)
    {
        cur = 1.0;
        for (int j = 0; j < n; j++)
        {
            cur *= double(min(i, arr[j])) / arr[j];;
        }
        res += (cur - pre) * i;
        pre = cur;
    }
    return res;
}


int main(int argc, const char *argv[])
{
    int n = 2;
    scanf("%d", &n);
    int *arr = new int [n]{}, MAX = 0;
    for (int i = 0; i < n; i++)
    {
        scanf("%d", &arr[i]);
        if (MAX < arr[i]) { MAX = arr[i]; }
    }
    printf("%.2lf\n", calProb(arr, n, MAX));
    delete []arr;
    return 0;
}
{% endhighlight %}

<p align="justify">
二维表第k大数<br>
在一块长为n，宽为m的场地上，有n✖️m个1✖️1的单元格。每个单元格上的数字就是按照从1到n和1到m中的数的乘积。具体如下<br>
n = 3, m = 3<br>
1   2   3<br>
2   4   6<br>
3   6   9<br>
给出一个查询的值k，求出按照这个方式列举的的数中第k大的值v。<br>
例如上面的例子里，<br>
从大到小为(9, 6, 6, 4, 3, 3, 2, 2, 1)<br>
k = 1, v = 9<br>
k = 2, v = 6<br>
k = 3, v = 6<br>
...<br>
k = 8, v = 2<br>
k = 9, v = 1<br><br>

<b>输入描述:</b><br>
只有一行是3个数n, m, k 表示场地的宽高和需要查询的k。使用空格隔开。<br><br>

<b>输出描述:</b><br>
给出第k大的数的值。<br><br>

<b>示例:</b><br>
输入<br>
3 3 4<br>
输出<br>
4<br><br>

<b>备注:</b><br>
【数据范围】<br>
100%的数据<br>
1 <= n, m <= 40000<br>
1 <= k <= n * m<br>
30%的数据<br>
1 <= n, m <= 1000<br><br>

<b>Solution:</b>
</p>
{% highlight C++ %}
#include<iostream>
using namespace std;

long getKthNum(long n, long m, long k)
{
    long left = 1, right = n * m;
    k = n * m - k + 1;
    while (left < right)
    {
        long mid = (left + right) / 2;
        long nRow = mid / m, count = nRow * m;
        for (long i = nRow + 1; i <= n; i++) { count += mid / i; }
        if (count < k) { left = mid + 1; }
        else { right = mid; }
    }
    return left;
}

int main(int argc, const char * argv[])
{
    long n = 3, m = 3, k = 4;
    scanf("%ld%ld%ld", &n, &m, &k);
    printf("%ld\n", getKthNum(n, m, k));
    return 0;
}
{% endhighlight %}


## 2. Java
<p align="justify">
<b>How to call a function?</b><br>
$\bigstar$ How to call a function in the same package?<br>
Consider we have a package called <b>P</b>, which have two files <b>file1.java</b> and <b>file2.java</b>.<br><br>

In file1, we want to import file2
</p>
{% highlight Java %}
Package P
import P.file2
public class file1 {
	public static void main(String[] args) {
		// TODO Auto-generated method stub
	}
}
{% endhighlight %}

<p align="justify">
$\bigstar$ How to call a function in another package?<br>
If I have two packages called <b>P1</b> and <b>P2</b>, P1 has a file <b>file1.java</b> and P2 has a file <b>file2.java</b><br><br>

In file1, we want to import file2
</p>
{% highlight Java %}
Package P1
import P2.file2
public class file1 {
	public static void main(String[] args) {
		// TODO Auto-generated method stub
	}
}
{% endhighlight %}

## 3. Python
<p align="justify">
<b>How to install un package in Anaconda by setup.py</b><br>
conda list<br>
/anaconda/lib/site-packages<br>
cd enter the directory site-packages<br>
python setup.py install<br><br>

<b>In fun(*args,**kwargs), what do *args and **kwargs mean?</b><br>
*args and **kwargs can be used as a container for infinitive arguments, namely, an uncertain number of arguments. In details, *args is for values without keys or variable names; while **kwargs is for key-value arguments.
</p>
{% highlight Python %}
def fun(*args):
    for x in args:
        print(x)

fun(1, 'abc', [1, 2, 3])
#Output
#1
#abc
#[1, 2, 3]

def fun(**kwargs):
    for k, v in kwargs.items():
        print(k, v)

fun(a1=1, a2='abc', a3=[1, 2, 3])
#Output
#a1 1
#a2 abc
#a3 [1, 2, 3]
{% endhighlight %}

<p align="justify">
<b>How to inherit a parent class's __init__?</b><br>
Two ways:<br>
</p>
{% highlight Python %}
class Parent:
    def __init__(self, age=30):
        self.age = age

    def printAge(self):
        print('{} years old'.format(self.age))


class Child1(Parent):
    def __init__(self, age=10, name='Bob'):
        Parent.__init__(self, age=age)
        self.name = name

    def infos(self):
        print('{} is {} years old'.format(self.name, self.age))


class Child2(Parent):
    def __init__(self, age=5, name='Julie'):
        super(Child2, self).__init__(age=age)
        self.name = name

    def infos(self):
        print('{} is {} years old'.format(self.name, self.age))


if __name__ == '__main__':
    c1 = Child1()
    c1.infos()
    
    c2 = Child2()
    c2.infos()

#Output:
#Bob is 10 years old
#Julie is 5 years old
{% endhighlight %}

<p align="justify">
<b>How to merge 2 dicts? How to delete one key-value in a dict</b><br>
</p>
{% highlight Python %}
d1 = {0: '0', 1: '1'}
d2 = {2: '2', 3: '3'}
# Merge 2 dict
d1.update(d2)
# Delete one key-value
del d1[0]
{% endhighlight %}

<p align="justify">
<b>What is GIL?</b><a href="https://www.zhihu.com/question/25532384"> [1]</a><br>
GIL is Global Interpreter Lock. In a process, there might be more than 1 thread. If one thread is running, GIL orders the other threads wait until the running thread is finished. But it is possible that several process are running at a same time.<br>
Thread is a unit of CPU scheduling; Process is a unit of computing ressources. We can regard a process as a train, a thread as a wagon.<br>
A process has mutiple threads. $\rightarrow$ A train has mutilple wagons.<br>
Different process have different data. $\rightarrow$ Different trains have different passengers.<br>
At a same process, data is shared among different threads. $\rightarrow$ At a same train, one passenger can move from one wagon to another.<br>
One process cannot influence another, but if a thread breaks down, the process is over. $\rightarrow$ One train cannot disturb another train.
</p>
{% highlight Python %}

{% endhighlight %}

<p align="justify">
<b>What is decorator?</b><a href="https://foofish.net/python-decorator.html"> [2]</a><br>
</p>
{% highlight Python %}

{% endhighlight %}


<p align="justify">
<b>How ?</b><br>
</p>
{% highlight Python %}

{% endhighlight %}