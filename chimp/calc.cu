/** Simple host-side calculator program to test chimpz
 *  @author Aaron Moss
 */
#include <iostream>
#include <map>
#include <string>
#include <sstream>

#include "chimp.cuh"

/** Operator types for calculator */
enum oper {
	asn,	/**< assignment */
	
	eq,		/**< equality */
	ne,		/**< inequality */
	lt,		/**< less-than */
	le,		/**< less-than-or-equal */
	gt,		/**< greater-than */
	ge,		/**< greater-than-or-equal */
	
	add,	/**< addition */
	adda,	/**< assignment-addition */
	sub,	/**< subtraction */
	suba,	/**< assignment-subtraction */
	mul,	/**< multiplication */
	mula,	/**< assignment-multiplication */
	dvd,	/**< division */
	dvda,	/**< assignment-division */
	
	band,	/**< bitwise and */
	banda,	/**< assignment-bitwise and */
	bor,	/**< bitwise or */
	bora,	/**< assignment-bitwise or */
	bxor,	/**< bitwise xor */
	bxora,	/**< assignment-bitwise xor */
	lsh,	/**< left-shift */
	lsha,	/**< assignment-left-shift */
	rsh,	/**< right-shift */
	rsha,	/**< assignment-right-shift */
	
	sign,	/**< sign operator */
	size,	/**< size operator */
	limbs,	/**< limbs operator */
	
	gcd,	/**< gcd operator */
	
	none	/**< invalid operator */
};

oper getOper(std::string s) {
	using std::string;
	
	if ( s == string("=") ) return asn;
	
	else if ( s == string("==") ) return eq;
	else if ( s == string("!=") ) return ne;
	else if ( s == string("<") ) return lt;
	else if ( s == string("<=") ) return le;
	else if ( s == string(">") ) return gt;
	else if ( s == string(">=") ) return ge;
	
	else if ( s == string("+") ) return add;
	else if ( s == string("+=") ) return adda;
	else if ( s == string("-") ) return sub;
	else if ( s == string("-=") ) return suba;
	else if ( s == string("*") ) return mul;
	else if ( s == string("*=") ) return mula;
	else if ( s == string("/") ) return dvd;
	else if ( s == string("/=") ) return dvda;
	
	else if ( s == string("&") ) return band;
	else if ( s == string("&=") ) return banda;
	else if ( s == string("|") ) return bor;
	else if ( s == string("|=") ) return bora;
	else if ( s == string("^") ) return bxor;
	else if ( s == string("^=") ) return bxora;
	else if ( s == string("<<") ) return lsh;
	else if ( s == string("<<=") ) return lsha;
	else if ( s == string(">>") ) return rsh;
	else if ( s == string(">>=") ) return rsha;
	
	else if ( s == string("=sign") ) return sign;
	else if ( s == string("=size") ) return size;
	else if ( s == string("=limbs") ) return limbs;
	
	else if ( s == string("gcd") ) return gcd;
	
	else return none;
}

bool doesAsn(oper op) {
	return 
		( op == asn 
		|| op == adda || op == suba || op == mula || op == dvda 
		|| op == banda || op == bora || op == bxora || op == lsha || op == rsha 
		|| op == sign || op == size || op == limbs );
}

int main(int argc, char** argv) {
	using namespace std;
	using namespace chimp;
	
	string line, s;
	chimpz res, op1, op2;
	map<string, chimpz> vars;
	string v1;
	oper op;
	
	while ( cin.good() ) {
		getline(cin, line);
		if ( cin.eof() ) break;
		if ( line == string("quit") ) break;
		
		istringstream in(line);
		
		while ( in.good() ) {
			in >> s;
			if ( s[0] == '$' ) {
				//variable
				op1 = vars[s];
				v1 = s;
			} else {
				//value
				op1 = chimpz(s);
				v1 = string("$_");
			}
			
			if ( in.eof() ) {
				vars[v1] = op1;
				cout << v1 << " = " << (string)vars[v1] << endl;
				break;
			}
			
			in >> s;
			op = getOper(s);
			if ( op == none ) {
				cout << "Invalid operator `" << s << "'" << endl;
				break;
			}
			if ( in.eof() ) {
				cout << "Missing operator: format <op1> <op> <op2>" << endl;
				break;
			}
			
			in >> s;
			if ( s[0] == '$' ) {
				//variable
				op2 = vars[s];
			} else {
				//value
				op2 = chimpz(s);
			}
			
			switch( op ) {
				case asn: res = (op1 = op2); break;
				
				case eq: res = chimpz((int)(op1 == op2)); break;
				case ne: res = chimpz((int)(op1 != op2)); break;
				case lt: res = chimpz((int)(op1 < op2)); break;
				case le: res = chimpz((int)(op1 <= op2)); break;
				case gt: res = chimpz((int)(op1 > op2)); break;
				case ge: res = chimpz((int)(op1 >= op2)); break;
				
				case add: res = (op1 + op2); break;
				case adda: res = (op1 += op2); break;
				case sub: res = (op1 - op2); break;
				case suba: res = (op1 -= op2); break;
				case mul: res = (op1 * op2); break;
				case mula: res = (op1 *= op2); break;
				case dvd: res = (op1 / op2); break;
				case dvda: res = (op1 /= op2); break;
				
				case band: res = (op1 & op2); break;
				case banda: res = (op1 &= op2); break;
				case bor: res = (op1 | op2); break;
				case bora: res = (op1 |= op2); break;
				case bxor: res = (op1 ^ op2); break;
				case bxora: res = (op1 ^= op2); break;
				case lsh: res = (op1 << (unsigned int)op2); break;
				case lsha: res = (op1 <<= (unsigned int)op2); break;
				case rsh: res = (op1 >> (unsigned int)op2); break;
				case rsha: res = (op1 >>= (unsigned int)op2); break;
				
				case sign: res = (op1 = chimpz(op2.sign())); break;
				case size: res = (op1 = chimpz(op2.size())); break;
				case limbs: res = (op1 = chimpz(op2.limbs())); break;
				
				case gcd: res = chimpz::gcd(op1, op2); break;
			}
			
			if ( doesAsn(op) ) {
				vars["$_"] = res;
				vars[v1] = res;
				cout << v1 << " = " << (string)res << endl;
			} else {
				vars["$_"] = res;
				cout << "$_ = " << (string)res << endl;
			}
		}
	}
}
