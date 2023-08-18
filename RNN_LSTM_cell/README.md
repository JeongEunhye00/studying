# RNN, LSTM cell 구현
## RNN
- $h_t = tanh(W_x x_t + W_h h_{t-1} + b)$
- $y_t = f(W_y h_t + b)$

## LSTM
- $f_t = \sigma(W_f \ \cdot \ [h_{t-1}, x_t] \ + \ b_i)$
- $i_t = \sigma(W_i \ \cdot \ [h_{t-1}, x_t] \ + \ b_f)$
- $\tilde{C}_t = tanh(W_C \ \cdot \ [h_{t-1}, x_t] \ + \ b_C)$
- $C_t = f_t \times C_{t-1} \ + \ i_t \times \tilde{C}_t$
- $o_t = \sigma(W_o \ \cdot \ [h_{t-1}, x_t] \ + \ b_o)$
- $h_t = o_t \times tanh(C_t)$

### gate
- forget gate → $f_t$ : 이전 time step의 cell state를 얼마나 반영할지 결정
- input gate → $i_t$ : 현재 time step의 cell state에 새로운 정보를 얼마나 반영할지 결정
- update → $C_t$ : forget gate와 input gate의 출력을 기반으로 새로운 cell state 생성
- output gate → $o_t$ : 현재 time step의 hidden state를 만들기 위해 최종적으로 얻어진 cell state의 값을 얼마나 반영할지 결정