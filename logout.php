<?php
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    setcookie('User', '', time() - 7200, '/');
}

header('Location: /index.php');
exit();
?>
