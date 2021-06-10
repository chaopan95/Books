---
layout: post
title:  "Database"
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


## 1. Model and Language
### 1.1 Introduction to database system
#### 1.1.1 Database
<p align="justify">
<b>Database is a collection of data with association relationship (table).</b><br><br>

Structued database: Relational database<br>
Unstructured database: image, music etc.<br><br>

<b>Terms in a relational database</b><br>
Suppose we have a table (relation) called <b>Student GPA</b>
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DATAS/Database/1_1_1_1.png"/></center><br>
</p>

#### 1.1.2 Database System
<p align="justify">
Database system is a working environment for database. It has 5 components:<br>
$\bigstar$ DB: Database<br>
-- A set of tables<br>
$\bigstar$ DBMS: Database Management System<br>
-- A software for database, e.g. Oracle, Sybase, SQL server or MS Access<br>
$\bigstar$ DBAP: Database Application<br>
-- Applications specially developped for users based on DBMS<br>
$\bigstar$ DBA: Database Administrator<br>
-- Create database, use DBMS<br>
$\bigstar$ Basic Computer System<br>
-- Include I/O
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DATAS/Database/1_1_2_1.png"/></center>
</p>
<p align="justify">
<b>DBMS provides somes functions at the view of users:</b><br>
$\bigstar$ DDL: Data Definition Language<br>
-- Users can use DDL to define table / relation<br>
$\bigstar$ DML: Data Manipulation Language<br>
-- Users can use DML to add, delete, modify and query etc.<br>
$\bigstar$ DCL: Data Control Language<br>
-- Different users have different access permission.<br><br>
</p>

#### 1.1.3 DBMS Standard Structure
<p align="justify">
<b>DBMS has 3 levels:</b><br>
$\bigstar$ External level = User level<br>
-- data visualization<br>
$\bigstar$ Conceptual level = Logical level<br>
-- data management<br>
$\bigstar$ Interbal level = Physical level<br>
-- data stockage, index<br><br>

<b>Schema & View</b><br>
$\bigstar$ Schema: a structural description of data<br>
$\bigstar$ View: data in a shcema<br><br>

<b>3 schemas and 2 mappings</b><br>
$\bigstar$ 3 schemas: External Schema, Conceptual Schema, Internal Schema<br>
$\bigstar$ 2 mappings: E-C (external to conceptual) mapping, C-I (conceptual to internal) mapping<br><br>

Logical independence means when Conceptual Schema is changed, we noly modify E-C mapping instead of External Schema; Physical independence means when Conceptual Schema is chaned, we only modify C-I mapping instead of Internal Schema.<br><br>
</p>

#### 1.1.4 Classical Data Model
<p align="justify">
Relational model: Table<br>
Hierarchical model: Tree<br>
Network model: Graph<br><br>
</p>

### 1.2 Relational model
#### 1.2.1 Concepts
<p align="justify">
A relation is a table. A relational model is compose of<br>
$\bigstar$ basic structure (relation)<br>
$\bigstar$ operators<br>
-- basic: $\cup$ (UNION), $-$ (DIFFERENCE), $\times$ (PRODUCT), $\sigma$ (SELECT), $\pi$ (PROJECT)<br>
-- extended: $\cap$ (INTERSECT), $\bowtie$ (JOIN), $\div$ (DIVISION)<br>
$\bigstar$ integrity constraints.<br>
-- Entity integrity, Referential integrity, User-defined integrity<br><br>

<b>Table definition</b><br>
$\bigstar$ Domain: a set of values with same data type<br>
$\bigstar$ Cardinality: the number of values in a domain<br>
$\bigstar$ Cartesian Product: a full combination for a group of domians
$$D_{1} \times D_{2} \times \cdots \times D_{n} = \{ (d_{1}, d_{2}, \cdots, d_{n}) \mid d_{i} \in D_{i}, i = 1, 2, ..., n \}$$
-- If we have n domians and domain $D_{i}$ has $m_{i}$ values ($i = 1, 2, ..., n$),  $\prod_{i=1}^{n} m_{i}$ is the carnality of a cartesian product.<br>
$\bigstar$ Tuple: each element $(d_{1}, d_{2}, \cdots, d_{n})$ is called (n-)tuple<br>
$\bigstar$ Each value in a n-tuple is called component<br>
For example, we have a table<br>
<table class="c">
  <center><h4>Table 1: Student GPA</h4></center>
  <tr><th>Class</th><th>Course</th><th>Teacher</th><th>Semester</th><th>Student ID</th><th>Student Name</th><th>Grade</th></tr>
  <tr><td>981101</td><td>Database</td><td>Tom</td><td>2020 Fall</td><td>98110101</td><td>S1</td><td>100</td></tr>
  <tr><td>981101</td><td>Database</td><td>Tom</td><td>2020 Fall</td><td>98110102</td><td>S2</td><td>90</td></tr>
  <tr><td>981101</td><td>Database</td><td>Tom</td><td>2020 Fall</td><td>98110103</td><td>S3</td><td>80</td></tr>
  <tr><td>981102</td><td>CS</td><td>Edie</td><td>2020 Fall</td><td>98110101</td><td>S1</td><td>80</td></tr>
  <tr><td>981102</td><td>CS</td><td>Edie</td><td>2020 Fall</td><td>98110102</td><td>S2</td><td>50</td></tr>
</table>
</p>
<p align="justify">
(981101, Database, Tom, 2020 Fall, 98110101, S1, 100) is a 7-tuple<br>
$\bigstar$ Relation: a subset of a Cartesian Product<br>
$\bigstar$ Relational Schema: $R(A_{1}:D_{1}, A_{2}:D_{2}, \cdots, A_{n}:D_{n})$, $A_{i}$ is a attribute in a table, $D_{i}$ is the corresponded domain, n is called degree.<br>
$\bigstar$ First normal form: indivisible attribute or atomic value.<br>
$\bigstar$ Candidate Key: an attribute or a group of attributes uniquely discriminate different tuples/rows/records, e.g. Student ID is a good candidate key.<br>
$\bigstar$ Primary Key: slelect one among many candidate keys.<br>
$\bigstar$ Foreign Key: Suppose two tables R and S, a candidate key K is the primary key in S, but not a primary key in R. K is a foreign key of R. Because K connects R and S.<br><br>

<b>Integrity Constraints</b><br>
$\bigstar$ Entity integrity<br>
-- Primary Key cannot be none.<br>
-- None: absent value or uncertain value or nonsense<br>
$\bigstar$ Referential integrity<br>
-- R1 has a Foreign Key Fk, which is a Promary Key in R2. Fk can be eiter non or some value among R2's Pk. Other conditions are prohibited.<br>
$\bigstar$ User-defined integrity<br>
-- Users have to obey all rules in DBMS, e.g. data type, data range etc.<br><br>
</p>

#### 1.2.2 Relational algebra
<p align="justify">
<table class="c">
  <center><h4>Table 2: Relational algebra operations</h4></center>
  <tr><th>Operations</th><th>Table 1</th><th>Table 2</th><th>Representation</th></tr>
  <tr><td>UNION</td><td>R</td><td>S</td><td>$R \cup S$</td></tr>
  <tr><td>INTERSECTION</td><td>R</td><td>S</td><td>$R \cap S$</td></tr>
  <tr><td>DIFFERENCE</td><td>R</td><td>S</td><td>$R - S$</td></tr>
  <tr><td>Cartesian PRODUCT</td><td>R</td><td>S</td><td>$R \times S$</td></tr>
  <tr><td>PROJECT</td><td>R</td><td></td><td>$\pi_{A}(R)$</td></tr>
  <tr><td>SELECT</td><td>R</td><td></td><td>$\sigma_{Condi}(R)$</td></tr>
  <tr><td>JOIN</td><td>R</td><td>S</td><td>$R \underset{A \text{ } \theta \text{ } B}{\bowtie} S$</td></tr>
  <tr><td>DIVISION</td><td>R</td><td>S</td><td>$R \div S$</td></tr>
</table><br>
</p>
<p align="justify">
<b>UNION</b><br>
$$R \cup S = \{t \mid t \in R \vee t \in S\}, \quad \text{where t is a tuple}$$
$\bigstar$ The final set has no duplicated elements.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DATAS/Database/1_2_2_1.png"/></center><br>
</p>
<p align="justify">
<b>DIFFERENCE</b><br>
$$R - S = \{ t \mid t \in R \wedge t \notin S \}$$
$\bigstar$ R - S and S - R is different.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DATAS/Database/1_2_2_2.png"/></center><br>
</p>
<p align="justify">
<b>Cartesian PRODUCT</b><br>
$$R \times S = \{ \left \langle a_{1}, a_{2}, ..., a_{n}, b_{1}, b_{2}, ..., b_{m} \right \rangle \mid \left \langle a_{1}, a_{2}, ..., a_{n} \right \rangle \in R \wedge \left \langle b_{1}, b_{2}, ..., b_{m} \right \rangle \in S \}$$
where $\left \langle a_{1}, a_{2}, ..., a_{n} \right \rangle$ represent a tuple.<br>
If S has $n_{1}$ rows and $m_{1}$ degree; while R has $n_{2}$ rows and $m_{2}$ degree, so $S \times R$ has $n_{1} \times n_{2}$ rows and $m_{1} + m_{2}$ degree.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DATAS/Database/1_2_2_3.png"/></center><br>
</p>
<p align="justify">
<b>SELECT</b><br>
$$\sigma_{\text{Condi}}(R) = \{ t \mid t \in R \wedge \text{Condi}(t) = \text{true} \}$$
$\bigstar$ Priority: (), $\theta$, $\neg$, $\wedge$, $\vee$<br>
\theta \in \{ >, <, \leq, \geq, \neq, = \}
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DATAS/Database/1_2_2_4.png"/></center><br>
</p>
<p align="justify">
<b>PROJECT</b><br>
$$\Pi_{A_{i1}, A_{i2}, ..., A_{ik}} (R) = \{ \left \langle t[A_{i1}], t[A_{i2}], ..., t[A_{ik}] \right \rangle \mid t \in R \}$$
$\bigstar$ Resort the attribute according the project order<br>
$\bigstar$ Delet duplicated rows
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DATAS/Database/1_2_2_5.png"/></center><br>
</p>
<p align="justify">
<b>INTERSECTION</b><br>
$$R \cap S = \{ t \mid t \in R \wedge t \in S \}$$
$\bigstar$ INTERSECTION can be derived by DIFFERENCE
$$R \cap S = R - (R - S) = S - (S - R)$$
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DATAS/Database/1_2_2_6.png"/></center><br>
</p>
<p align="justify">
<b>$\theta$-JOIN</b><br>
$$R \underset{A \text{ } \theta \text{ } B}{\bowtie} S = \sigma_{t[A] \text{ } \theta \text{ } s[B]} (R \times S), \quad \text{where t is a tuple of R and s is a tuple of S}$$
$\bigstar$ Rename: $\rho_{\text{name2}} (\text{name1})$<br>
-- when we join one table and itself, we need rename
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DATAS/Database/1_2_2_7.png"/></center><br>
</p>
<p align="justify">
<b>Natural JOIN</b><br>
$$R \bowtie S = \sigma_{t[B] = s[B]} (R \times S)$$
$\bigstar$ Same attributes and same values
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DATAS/Database/1_2_2_8.png"/></center><br>
</p>
<p align="justify">
<b>DIVISION</b><br>
Precondition: relation $R(A_{1}, A_{2}, ..., A_{n})$ and relation S(B_{1}, B_{2}, ..., B_{m}) do $R \div S$ is possible if and only if $\{B_{1}, B_{2}, ..., B_{m}\} \subset \{ A_{1}, A_{2}, ..., A_{n} \}$, namely m < n.
$$
\begin{aligned}
R \div S & = \{t \mid t \in \Pi_{R-S}(R) \wedge \forall u \in S (tu \in R) \} \\
& = \Pi_{R-S} (R) - \Pi_{R-S} ((\Pi_{R-S}(R) \times S) - R)
\end{aligned}
$$
$\bigstar$ If R has m degree, S has n degree, $R \div S$ has m-n degree.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DATAS/Database/1_2_2_9.png"/></center><br>
</p>
<p align="justify">
<b>Outer JOIN</b><br>
$\bigstar$ Left Outer Join A ⟕ B<br>
$\bigstar$ Right Outer Join A ⟖ B<br>
$\bigstar$ Full Outer Join A ⟗ B<br>
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DATAS/Database/1_2_2_10.png"/></center><br>
</p>

#### 1.2.3 Relational calculus
<p align="justify">
<b>Relational tuple calculus</b><br>
Basic form
$$\{ t \mid P(t) \}, \quad \text{where } P(t) \text{ is an expression}$$
Three atomic expressions
$$
\begin{aligned}
& s \in R \\
& s[A] \text{ } \theta \text{ } c, \quad \text{where } \theta \in \{ <, > , \leq, \geq, \neq, =\} \\
& s[A] \text{ } \theta \text{ } u[B]
\end{aligned}
$$
Operator priority
$$(), \theta, \exists, \forall, \neg, \wedge, \vee$$

For example, in table 1 we want to query all records which selecting course Database and its grade is greater than 85.
$$\{ t \mid t \in \text{Table1} \wedge t[\text{Course} ] = \text{'Database'} \wedge t[\text{Grade} ] \geq 85 \}$$
We want to query all students which select Database
$$\{ t \mid \exists (t \mid t \in \text{Table1} ) (t[\text{Course} ] = \text{'Database'}) \}$$

<b>Relational domain calculus</b><br>
Basic form
$$\{ \left \langle x_{1}, x_{2}, ..., x_{n} \right \rangle \mid P(x_{1}, x_{2}, ..., x_{n}) \}$$
Three atomic expressions
$$
\begin{aligned}
& \left \langle x_{1}, x_{2}, ..., x_{n} \right \rangle \in R \\
& x \text{ } \theta \text{ } c \\
& x \text{ } \theta \text{ } y
\end{aligned}
$$

For example, we want to query all students which select Database
$$\{ \left \langle \text{a, b, c, d, e, f, g} \right \rangle \mid \left \langle \text{a, b, c, d, e, f, g} \right \rangle \in \text{Table1} \wedge b = \text{Database} \}$$
</p>

### 1.3 SQL
#### 1.3.1 DLL
<p align="justify">
Suppose we have 5 tables: student, departement, course, teacher, select course<br>
Student (S# char(8), Sname char(10), Ssex char(2), Sage integer, D#, char(2), Sclass char(6))<br>
Dept (D# char(2), Dname char(10), Dean char(10))<br>
Course (C# char(3), Cname char(12), Chours integer, Credit float(1), T# char(3))<br>
Teacher (T# char(3), Tname char(10), D# char(2), Salary float(12))<br>
SC (S# char(8), C# char(3), Score float(1))<br><br>

<b>Create</b><br>
Create dtabase SCT
</p>
{% highlight SQL %}
Create database SCT;
{% endhighlight %}
<p align="justify">
Create table
</p>
{% highlight SQL %}
Create Table student (S# char(8), Sname char(10), Ssex char(2),
                      Sage integer, D#, char(2), Sclass char(6))
{% endhighlight %}
<p align="justify">
<b>Alter</b><br>
For example, add 2 columns Saddr, PID in Student (S#, Sname, Ssex, Sage, D#, Sclass)
</p>
{% highlight SQL %}
Alter Table Student Add Saddr[40], PID, char[18];
{% endhighlight %}
<p align="justify">
Change Sname's property in Student with 10 char
</p>
{% highlight SQL %}
Alter Table Student Modify Sname char(10);
{% endhighlight %}
<p align="justify">
<b>Drop</b><br>
Delete a table/database
</p>
{% highlight SQL %}
Drop Table Student;
Drop Table teacher;
Drop database SCT;
{% endhighlight %}
<p align="justify">
</p>

#### 1.3.2 DML
<p align="justify">
<b>Select</b><br>
Query all informations in student table
</p>
{% highlight SQL %}
Select S#, Sname, Ssex, Sage, Sclass, D#
From Student;

Select * From student;
{% endhighlight %}
<p align="justify">
Query all student names and student age, where the students are less 18-years-old
</p>
{% highlight SQL %}
Select Sage, Sname
From Student
Where Sage < 18;
{% endhighlight %}
<p align="justify">
Query all student name where they have learned course 001 or course 002
</p>
{% highlight SQL %}
Select S# From SC
Where C# = '001' OR C# = '002';
{% endhighlight %}
<p align="justify">
$\bigstar$ DISTINCT denotes remove all duplicated rows, e.g. query all student ID where the students' grade is bigger than 80.
</p>
{% highlight SQL %}
Select DISTINCT S#
From SC
Where Score > 80;
{% endhighlight %}
<p align="justify">
$\bigstar$ Order By ASC/DESC, e.g. query all students' ID and these students select course 002 and their grades are bigger than 80. The result is in descendent order with regards to score.
</p>
{% highlight SQL %}
Select S# From SC Where C# = '002' and Score > 80 Order By Score;
{% endhighlight %}
<p align="justify">
$\bigstar$ (Not) Like (Fuzzy Search) e.g. query all students' ID and name where their names doesn't have Tom at first
</p>
{% highlight SQL %}
Select S#, Sname From Student
Where Sname Not Like 'Tom%';
{% endhighlight %}
<p align="justify">
$\bigstar$ Multiple-table query e.g. query student names where the students select course 001. The result is in descent order with respect to grade
</p>
{% highlight SQL %}
Select Sname From Student, SC
Where Student.S# = SC.S# and SC.C# = '001'
Order By Score DESC
{% endhighlight %}
<p align="justify">
$\bigstar$ Renmae e.g. query all teachers where each two teacher have different salaries
</p>
{% highlight SQL %}
Select T1.Tname as Teacher1, T2.Tname as Teacher2
From Teacher T1, Teacher T2
Where T1.Salary > T2.Salary;
{% endhighlight %}
<p align="justify">
Query all students which select both course 001 and course 002, return their ID
</p>
{% highlight SQL %}
Select S1.S# From SC SC1, SC SC2
Where SC1.S# = SC2.S# and SC1.C# = '001' and SC2.C# = '002';
{% endhighlight %}
<p align="justify">
<b>Insert</b><br>
</p>
{% highlight SQL %}
Insert into Student
Values ('98030101', 'Tom', 'Male', 20, '03', '980301');

Insert into Student (S#, Sname, Ssex, Sage, D#, Sclass)
Values ('98030102', 'Julie', 'Female', 19, '03', '980301');
{% endhighlight %}
<p align="justify">
Create a new table St(S#, Sname), insert what we query into the table
</p>
{% highlight SQL %}
Create Table St (S#, Sname);
Insert into St (S#, Sname)
  Select S#, Sname From Student Where Sname Like 'Tom%';
Insert into St (S#, Sname)
  Select S#, Sname From Student Order By Snmae;
{% endhighlight %}
<p align="justify">
<b>Delete</b><br>
$\bigstar$ Delete all rows
</p>
{% highlight SQL %}
Delete From SC;
Delete From SC Where S# = '98030101';
Delete From Student Where D# is
  (Select D# From Dept Where Dname = 'Computer Science');
{% endhighlight %}
<p align="justify">
<b>Update</b><br>
For example, augment salary of teacher by 5%
</p>
{% highlight SQL %}
Update Teacher
Set Salary = Salary * 1.05;
{% endhighlight %}
<p align="justify">
Augment salary of teachers in Computer Science by 10%
</p>
{% highlight SQL %}
Update Teacher
Set Salary = Salary * 1.1
Where D# in (Select D# From Dept Where Dname = 'Computer Science');
{% endhighlight %}

#### 1.3.3 DCL
<p align="justify">
<b>Grant</b><br>

<b>Revoke</b><br>
</p>


## 2. Database Design
### 2.1 Idea and Method
#### 2.1.1 Introduction
<p align="justify">

</p>


## 3. Realization
<p align="justify">
How to stock data efficiently?<br>
How to query data efficiently?
</p>

### 3.1 Stockage
#### 3.1.1 Two main tasks
<p align="justify">
Physical stockage:<br>
1) reduce I/O times<br>
2) reduce waiting time
</p>

#### 3.1.2 Magnetic disk
<p align="justify">

</p>

#### 3.1.3 File organization
<p align="justify">
$\bigstar$ Unorder file: Heap file or pile file<br>
-- fast to insert/delete/modify<br>
$\bigstar$ Sequential file: ordering key<br>
-- query fast but insert is not sure to be fast<br>
-- solution: overflow file or reorganization<br>
$\bigstar$ Hash file<br>
$\bigstar$ Cluster file<br>
-- cluster according to a smae attribute<br><br>

If the performance is less and les sufficient with time, we should consider reorganize our database.
</p>

### 3.2 Index
#### 3.2.1 Introduction
<p align="justify">
Index is an auxiliary structure for tables, composed of <b>Index-field</b> and <b>pointer</b>. Index file improve query performance for the main files.<br><br>

2 kinds of index file: ordered index file and hash index file.<br><br>

Dense index and Sparse index<br><br>

B+ tree index<br>
Leaf nodes piont to main file; not leaf nodes point to index.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DATAS/Database/3_2_1_1.png"/></center><br>
</p>

### 3.3 Algorithm
#### 3.3.1 Sort
<p align="justify">
$\bigstar$ Internal sort<br>
$\bigstar$ External sort<br>
-- divided into several small subset which can be loaded into memory<br>
-- sort each subset then write into disk<br>
-- multi-way merge sort
</p>

#### 3.3.2 Hash
<p align="justify">

</p>

#### 3.3.2 B+
<p align="justify">

</p>

### 3.4 Optimization
#### 3.4.1 General operations
<p align="justify">
$\bigstar$ Semantic optimization<br>
-- Integrity Constraints<br>
-- DBMS detect and analyse automatically which tables or which attributes are useless then ignore them<br>
$\bigstar$ Grammar optimization or logical optimization<br>
-- DBMS convert a SQL into a relational algebra<br>
-- Different algebra orders have different efficiency.<br>
-- It whould be better to perform select/project operation for sake of efficiency<br>
-- To prove 2 operations can be switched<br>
$\bigstar$ Operation optimization<br>
-- stockage and operation<br>
-- choice of algorithms<br>
-- cost estimation<br>
-- operation order
</p>

#### 3.4.2 Logical operation
<p align="justify">
Several policies<br>
$\bigstar$ select and project as early as possible<br>
$\bigstar$ combine select and product
</p>

#### 3.4.3 Relational algebra equality
<p align="justify">
$$
\begin{aligned}
& E_{1} \bowtie E_{2} \equiv E_{2} \bowtie E_{1} \\
& E_{1} \times E_{2} \equiv E_{2} \times E_{1} \\
& (E_{1} \bowtie E_{2}) \bowtie E_{3} = E_{1} \bowtie (E_{2} \bowtie E_{3}) \\
& \sigma_{F_1} (\sigma_{F_{2}}(E)) \equiv \sigma_{F_{1} \wedge F_{2}} (E) \\
& \Pi_{A_{1}, A_{2}, ..., A_{n}} (\sigma_{F} (E)) \equiv \sigma_{F} (\Pi_{A_{1}, A_{2}, ..., A_{n}} (E)) \\
& \Pi_{A_{1}, ..., A_{n}} (E_{1} \cup E_{2}) \equiv \Pi_{A_{1}, ..., A_{n}} (E_{1}) \cup \Pi_{A_{1}, ..., A_{n}} (E_{2})
\end{aligned}
$$
</p>

#### 3.4.3 Physical query optimization
<p align="justify">
TableScan(R)<br>
SortTableScan(R)<br>
IndexScan(R)<br>
SortIndexScan(R)
</p>

### 3.5 Transaction and Concurrence
#### 3.5.1 ACID
<p align="justify">
$\bigstar$ Atomicity<br>
-- either do all or do nothing<br>
$\bigstar$ Consistency<br>
$\bigstar$ Isolation<br>
-- two transactions are seperated<br>
$\bigstar$ Durability
</p>

#### 3.5.2 Schedule
<p align="justify">
$\bigstar$ Serialiable schedule<br>
$\bigstar$ Concurrent schedule<br><br>

$\bigstar$ Serializability<br>
-- to measure the correctness of schedule
</p>

<p align="justify">

</p>
{% highlight SQL %}

{% endhighlight %}