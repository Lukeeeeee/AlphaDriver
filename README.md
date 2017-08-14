# DDPG-Torcs-UV

### Observation
<table>
  <thead>
    <tr>
      <th style="text-align: left">Name</th>
      <th style="text-align: left">Range (units)</th>
      <th style="text-align: left">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left">ob.angle</td>
      <td style="text-align: left">[-<script type="math/tex">\pi</script>,+<script type="math/tex">\pi</script>]</td>
      <td style="text-align: left">Angle between the car direction and the direction of the track axis</td>
    </tr>
    <tr>
      <td style="text-align: left">ob.track</td>
      <td style="text-align: left">(0, 200)(meters)</td>
      <td style="text-align: left">Vector of 19 range finder sensors: each sensor returns the distance between the track edge and the car within a range of 200 meters</td>
    </tr>
    <tr>
      <td style="text-align: left">ob.trackPos</td>
      <td style="text-align: left">(-<script type="math/tex">\infty</script>,+<script type="math/tex">\infty</script>)</td>
      <td style="text-align: left">Distance between the car and the track axis. The value is normalized w.r.t. to the track width: it is 0 when the car is on the axis, values greater than 1 or -1 means the car is outside of the track.</td>
    </tr>
    <tr>
      <td style="text-align: left">ob.speedX</td>
      <td style="text-align: left">(-<script type="math/tex">\infty</script>,+<script type="math/tex">\infty</script>)(km/h)</td>
      <td style="text-align: left">Speed of the car along the longitudinal axis of the car (good velocity)</td>
    </tr>
    <tr>
      <td style="text-align: left">ob.speedY</td>
      <td style="text-align: left">(-<script type="math/tex">\infty</script>,+<script type="math/tex">\infty</script>)(km/h)</td>
      <td style="text-align: left">Speed of the car along the transverse axis of the car</td>
    </tr>
    <tr>
      <td style="text-align: left">ob.speedZ</td>
      <td style="text-align: left">(-<script type="math/tex">\infty</script>,+<script type="math/tex">\infty</script>)(km/h)</td>
      <td style="text-align: left">Speed of the car along the Z-axis of the car</td>
    </tr>
    <tr>
      <td style="text-align: left">ob.wheelSpinVel</td>
      <td style="text-align: left">(0,+<script type="math/tex">\infty</script>)(rad/s)</td>
      <td style="text-align: left">Vector of 4 sensors representing the rotation speed of wheels</td>
    </tr>
    <tr>
      <td style="text-align: left">ob.rpm</td>
      <td style="text-align: left">(0,+<script type="math/tex">\infty</script>)(rpm)</td>
      <td style="text-align: left">Number of rotation per minute of the car engine</td>
    </tr>
  </tbody>
</table>




### Action
<table>
  <thead>
    <tr>
      <th style="text-align: left">Action</th>
      <th style="text-align: left"><script type="math/tex">\theta</script></th>
      <th style="text-align: left"><script type="math/tex">\mu</script></th>
      <th style="text-align: left"><script type="math/tex">\sigma</script></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left">steering</td>
      <td style="text-align: left">0.6</td>
      <td style="text-align: left">0.0</td>
      <td style="text-align: left">0.30</td>
    </tr>
    <tr>
      <td style="text-align: left">acceleration</td>
      <td style="text-align: left">1.0</td>
      <td style="text-align: left">[0.3-0.6]</td>
      <td style="text-align: left">0.10</td>
    </tr>
    <tr>
      <td style="text-align: left">brake</td>
      <td style="text-align: left">1.0</td>
      <td style="text-align: left">-0.1</td>
      <td style="text-align: left">0.05</td>
    </tr>
  </tbody>
</table>

